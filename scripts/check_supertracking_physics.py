"""Offline BAM/MuJoCo-Warp audit: hold poses and replay a reference without a policy."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_openchicken.robot.openchicken_constants import OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG
from mjlab_openchicken.tasks.openchicken_velocity_env_cfg import make_openchicken_velocity_env_cfg
from prepare_supertracking_motion import geom_vertices, sha256, training_model


def audit(motion_path, output, device):
    with np.load(motion_path, allow_pickle=False) as raw:
        motion = {k: raw[k] for k in raw.files}
    cfg = make_openchicken_velocity_env_cfg(play=True)
    cfg.scene.entities = {"robot": deepcopy(OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG)}
    cfg.scene.sensors = ()
    cfg.scene.num_envs = 16
    cfg.observations = {}
    cfg.commands = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.events = {"expand_bam_friction_fields": cfg.events["expand_bam_friction_fields"]}
    cfg.episode_length_s = 11
    cfg.auto_reset = False
    cfg.seed = 42
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 1024
    # Nominal voltage and sag, with the actual training actuator's random delay.
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.vin_range = (7.4, 7.4)
    actuator.vin_drop_gain_range = (.1, .1)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        robot = env.scene["robot"]
        names = tuple(n for n in robot.joint_names if not n.startswith("passive_"))
        assert tuple(motion["joint_names"]) == names
        servo_ids = [robot.joint_names.index(n) for n in names]
        assert abs(env.step_dt - .02) < 1e-9
        env.reset()
        model, _, _ = training_model()
        data = mujoco.MjData(model)
        qadr = np.array([model.joint(n).qposadr[0] for n in names])
        home = model.qpos0.copy()
        home[qadr] = robot.data.default_joint_pos[0, servo_ids].cpu().numpy()
        data.qpos[:] = home
        mujoco.mj_forward(model, data)
        foot_ids = [model.geom(n).id for n in ("left_foot_collision", "right_foot_collision")]
        home[2] -= min(geom_vertices(model, data, g)[:, 2].min() for g in foot_ids)
        labels = ["home", "ready", "up", "recenter", "down"]
        reference_qpos = np.tile(model.qpos0, (len(motion["qpos"]), 1))
        reference_qpos[:, :7] = motion["qpos"][:, :7]
        reference_qpos[:, qadr] = motion["joint_pos"]
        targets = np.stack([home, *reference_qpos[[0, 16, 38, 56]]])
        starts = np.repeat(targets, 3, axis=0)
        rng = np.random.default_rng(42)
        for i in range(len(starts)):
            if i % 3:
                starts[i, qadr] += rng.normal(0, np.deg2rad(.5), len(qadr))
                rot = Rotation.from_rotvec(rng.normal(0, np.deg2rad(.5), 3))
                starts[i, 3:7] = (rot * Rotation.from_quat(starts[i, 3:7], scalar_first=True)).as_quat(scalar_first=True)
        starts = np.concatenate([starts, reference_qpos[:1]])
        origins = env.scene.env_origins.cpu().numpy()
        initial = starts.copy()
        initial[:, :3] += origins
        env.sim.data.qpos[:] = torch.tensor(initial, dtype=torch.float32, device=device)
        env.sim.data.qvel[:] = 0
        # Fill the actuator delay queue with the actual initial target.
        for act in robot.actuators:
            if act._delay_buffer is not None:
                act._delay_buffer.reset()
        env.sim.forward()
        robot.update(0)
        fixed_q = np.repeat(targets[:, qadr], 3, axis=0)
        poses, velocities, torques = [starts.copy()], [], []
        for frame in range(500):
            desired = np.concatenate([fixed_q, motion["joint_pos"][frame:frame + 1]])
            actions = torch.tensor(desired, dtype=torch.float32, device=device) - robot.data.default_joint_pos[:, servo_ids]
            env.step(actions)
            current = env.sim.data.qpos[:].cpu().numpy().copy()
            current[:, :3] -= origins
            poses.append(current)
            velocities.append(env.sim.data.qvel[:].cpu().numpy().copy())
            torques.append(env.sim.data.qfrc_actuator[:].cpu().numpy().copy())
            if frame % 50 == 49:
                print(f"Physics audit: {(frame + 1) / 50:.0f}/10 s", flush=True)
        poses, velocities, torques = map(np.asarray, (poses, velocities, torques))
        np.savez_compressed(output / "physics_rollouts.npz", qpos=poses, qvel=velocities,
                            generalized_actuator_force=torques, time=np.arange(501) / 50,
                            labels=np.array([f"{n}_seed{s}" for n in labels for s in range(3)] + ["v12_replay"]))
        cases = []
        for i in range(16):
            end = 151 if i < 15 else 501
            actual = poses[:end, i]
            matrices = Rotation.from_quat(actual[:, 3:7], scalar_first=True).as_matrix()
            tilt = np.rad2deg(np.arccos(np.clip(matrices[:, 2, 2], -1, 1)))
            fail = (tilt > 45) | (actual[:, 2] < .055) | ~np.isfinite(actual).all(axis=1)
            desired = targets[i // 3][None, :] if i < 15 else reference_qpos
            cases.append({
                "name": f"{labels[i // 3]}_seed{i % 3}" if i < 15 else "v12_replay",
                "duration_s": (end - 1) / 50, "max_tilt_deg": float(tilt.max()),
                "final_tilt_deg": float(tilt[-1]), "minimum_root_height_m": float(actual[:, 2].min()),
                "first_fall_time_s": float(np.flatnonzero(fail)[0] / 50) if fail.any() else None,
                "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean((actual[:, qadr] - desired[:, qadr]) ** 2)))),
                "root_position_rmse_mm": float(np.sqrt(np.mean(np.sum((actual[:, :3] - desired[:, :3]) ** 2, axis=1))) * 1000),
                "max_abs_actuator_torque_nm": float(abs(torques[:end - 1, i, 6:]).max()),
                "pose_hold_or_open_loop_pass": bool(not fail.any()),
            })
        report = {
            "motion_sha256": sha256(motion_path), "engine": "mjlab 1.3.0 / MuJoCo Warp",
            "device": device, "actuator": "unmodified FrictionDRBamActuator / XL330 M6",
            "physics_dt": env.physics_dt, "control_dt": env.step_dt,
            "voltage_v": 7.4, "voltage_sag_gain": .1, "minimum_voltage_v": actuator.vin_min,
            "delay_physics_steps": [actuator.delay_min_lag, actuator.delay_max_lag],
            "noise_std_deg": .5, "seed": 42, "cases": cases,
            "all_pose_holds_pass": all(c["pose_hold_or_open_loop_pass"] for c in cases[:-1]),
            "open_loop_replay_pass": cases[-1]["pose_hold_or_open_loop_pass"],
            "limitations": ["Open-loop joint targets, not a trained stabilizing policy.",
                            "A failed replay does not prove the motion cannot be learned.",
                            "Reference self-intersections must be resolved independently.",
                            "No robot hardware was accessed."],
        }
        (output / "physics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    audit(args.motion, args.output, args.device)
