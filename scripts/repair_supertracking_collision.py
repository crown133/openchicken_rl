"""Offline V12 repair: stagger the left foot without changing upper-body motion."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from prepare_supertracking_motion import differentiate, geom_label, sha256, training_model


def repair_cycle(model, names, cycle):
    """Move the fixed left stance +25 mm; preserve all other coordinates."""
    data = mujoco.MjData(model)
    joint_ids = [model.joint(n).id for n in names if n.startswith("left_")]
    addresses = model.jnt_qposadr[joint_ids]
    limits = model.jnt_range[joint_ids] + np.deg2rad([1, -1])
    foot = model.body("ankle_left").id
    fixed = cycle.copy()
    previous = None
    for i, pose in enumerate(cycle):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        target_pos = data.xpos[foot].copy() + [.025, 0, 0]
        target_rot = data.xmat[foot].reshape(3, 3).copy()

        def residual(angles):
            data.qpos[addresses] = angles
            mujoco.mj_kinematics(model, data)
            rotation_error = Rotation.from_matrix(
                data.xmat[foot].reshape(3, 3) @ target_rot.T
            ).as_rotvec()
            return np.r_[(data.xpos[foot] - target_pos) * 1000, rotation_error * 100]

        initial = np.clip(pose[addresses] if previous is None else previous,
                          limits[:, 0], limits[:, 1])
        result = least_squares(
            residual, initial,
            bounds=(limits[:, 0], limits[:, 1]), max_nfev=150,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        error = residual(result.x)
        if not result.success or np.max(np.abs(error)) > .001:
            raise RuntimeError(f"Foot IK failed at cycle frame {i}: {error}")
        fixed[i, addresses] = result.x
        previous = result.x.copy()
    return fixed


def repair(source, cycle_path, output):
    output.mkdir(parents=True, exist_ok=False)
    model, names, _ = training_model()
    data = mujoco.MjData(model)
    with np.load(source) as raw:
        reference = {key: raw[key] for key in raw.files}
    cycle = np.load(cycle_path)["qpos"]
    assert tuple(reference["joint_names"]) == names
    assert cycle.shape == (250, 7 + len(names))
    assert len(reference["time"]) == 501 and reference["fps"] == 50
    sample_indices = np.arange(501) * 3 % 250
    original = np.c_[reference["root_pos"], reference["root_quat_wxyz"], reference["joint_pos"]]
    assert np.allclose(original, cycle[sample_indices], atol=1e-12)
    joint_ids = [model.joint(n).id for n in names]
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    full_cycle = np.tile(model.qpos0, (len(cycle), 1))
    full_cycle[:, :7], full_cycle[:, qadr] = cycle[:, :7], cycle[:, 7:]
    fixed = repair_cycle(model, names, full_cycle)
    addresses = np.array([model.joint(n).qposadr[0] for n in names if n.startswith("left_")])
    poses = fixed[sample_indices]
    velocity = differentiate(model, poses, .02)
    reference["joint_pos"] = poses[:, qadr]
    reference["joint_vel"] = velocity[:, vadr]
    reference["kinematic_only"] = np.array(True)
    np.savez_compressed(output / "style_reference_50hz.npz", **reference)
    np.savez_compressed(output / "cycle_150hz.npz", time=np.arange(250) / 150, qpos=np.c_[fixed[:, :7], fixed[:, qadr]])

    # Audit the actual 50 Hz samples AND their subframes. This is the explicit
    # interpolation contract: linear positions/joints, shortest-path root SLERP.
    times = np.arange(6001) / 600
    dense = np.stack([np.interp(times, reference["time"], poses[:, j]) for j in range(model.nq)], axis=1)
    dense[:, 3:7] = Slerp(reference["time"], Rotation.from_quat(poses[:, 3:7], scalar_first=True))(times).as_quat(scalar_first=True)
    floor = model.geom("audit_floor").id
    soles = [model.geom(n).id for n in ("left_foot_collision", "right_foot_collision")]
    battery = next(g for g in range(model.ngeom) if geom_label(model, g) == "trunk_base/np_f970" and model.geom_contype[g])
    shins = [next(g for g in range(model.ngeom) if geom_label(model, g) == name and model.geom_contype[g])
             for name in ("leg/leg", "leg_2/leg")]
    vertices = {}
    for geom in [battery, *shins, *soles]:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start + count]
        vertices[geom] = points[ConvexHull(points).vertices]

    def world_vertices(geom):
        return vertices[geom] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]

    # A positive projection gap certifies separation of the collision convex
    # hulls. It is a lower bound on distance, not an assumed nearest-point result.
    previous_axes = [np.array([1., 0, 0]), np.array([1., 0, 0])]
    clearance = [float("inf"), float("inf")]
    initial_soles = None
    sole_drift = 0.
    sole_height = float("inf")
    contacts = []
    for i, pose in enumerate(dense):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        sole_points = np.concatenate([world_vertices(g) for g in soles])
        if initial_soles is None:
            initial_soles = sole_points.copy()
        sole_drift = max(sole_drift, float(np.linalg.norm(sole_points - initial_soles, axis=1).max()))
        sole_height = min(sole_height, float(sole_points[:, 2].min()))
        a = world_vertices(battery)
        for j, shin in enumerate(shins):
            b = world_vertices(shin)
            segment = np.zeros(6)
            mujoco.mj_geomDistance(model, data, battery, shin, .1, segment)
            axes = np.vstack([segment[3:] - segment[:3], previous_axes[j], b.mean(0) - a.mean(0), np.eye(3)])
            lengths = np.linalg.norm(axes, axis=1)
            axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
            ap, bp = a @ axes.T, b @ axes.T
            gaps = np.maximum(bp.min(0) - ap.max(0), ap.min(0) - bp.max(0))
            best = int(np.argmax(gaps))
            previous_axes[j] = axes[best]
            clearance[j] = min(clearance[j], float(gaps[best]))
        for contact in data.contact:
            if contact.dist >= -.0001:
                continue
            if floor in (contact.geom1, contact.geom2) and (contact.geom1 in soles or contact.geom2 in soles):
                continue
            contacts.append({"time": float(times[i]), "pair": [geom_label(model, contact.geom1), geom_label(model, contact.geom2)], "depth_mm": float(-contact.dist * 1000)})
        if i % 1200 == 0:
            print(f"Audited {i}/{len(times)} interpolated samples", flush=True)
    all_ids = [model.joint(n).id for n in names]
    margin = np.minimum(dense[:, 7:] - model.jnt_range[all_ids, 0], model.jnt_range[all_ids, 1] - dense[:, 7:])
    report = {
        "source": str(source.resolve()), "source_sha256": sha256(source),
        "cycle_source_sha256": sha256(cycle_path),
        "repair": "Left foot fixed stance translated +25 mm on world X; only five left leg joints changed",
        "preserved_exactly": "root position/orientation, head/neck joints, right leg, 10 s timing, six repetitions",
        "interpolation": "linear root position/joints; root quaternion SLERP",
        "audit_hz": 600, "audit_samples": len(times),
        "battery_shin_certified_minimum_gap_mm": (np.array(clearance) * 1000).tolist(),
        "maximum_sole_vertex_drift_mm": sole_drift * 1000,
        "minimum_sole_height_mm": sole_height * 1000,
        "minimum_joint_limit_margin_deg": float(np.rad2deg(margin.min())),
        "maximum_left_joint_change_deg": float(np.rad2deg(np.abs(poses[:, addresses] - original[:, addresses])).max()),
        "maximum_joint_step_50hz_deg": float(np.rad2deg(np.abs(np.diff(poses[:, 7:], axis=0))).max()),
        "nonfoot_penetrations_above_0_1mm": contacts,
        "collision_pass": not contacts and min(clearance) > .002,
        "kinematic_only": True, "dynamic_balance_validated": False,
    }
    (output / "collision_report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["collision_pass"], report
    assert sole_drift < .00015 and sole_height > -.00015 and margin.min() > 0
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair(args.source, args.cycle, args.output)
