"""Map an offline OpenChicken reference to the training MJCF; never drive hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from mjlab.entity import Entity
from mjlab_openchicken.robot.openchicken_constants import (
    OPENCHICKEN_ALLCOLLISIONS_XML,
    OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_model():
    robot = Entity(OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG)
    robot.spec.worldbody.add_geom(
        name="audit_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, .01], contype=1, conaffinity=1,
    )
    joints = tuple(n for n in robot.joint_names if not n.startswith("passive_"))
    bodies = tuple(n for n in robot.body_names if n not in ("comb", "lower_beak"))
    return robot.spec.compile(), joints, bodies


def geom_vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("Sole geometry must be a mesh")
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def geom_label(model, geom):
    mesh = model.geom_dataid[geom]
    shape = model.mesh(mesh).name if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else str(model.geom_type[geom])
    return f"{model.body(model.geom_bodyid[geom]).name}/{model.geom(geom).name or shape}"


def differentiate(model, poses, dt):
    """Central qvel in MuJoCo tangent coordinates, wrapping the duplicate endpoint."""
    if not np.allclose(poses[0], poses[-1], atol=1e-8):
        raise ValueError("This converter expects a closed clip with duplicate endpoint")
    unique = poses[:-1]
    velocities = np.empty((len(poses), model.nv))
    for i in range(len(unique)):
        mujoco.mj_differentiatePos(model, velocities[i], 2 * dt,
                                 unique[(i - 1) % len(unique)], unique[(i + 1) % len(unique)])
    velocities[-1] = velocities[0]
    return velocities


def prepare(source, preview_xml, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source, allow_pickle=False) as raw:
        reference = {key: raw[key] for key in raw.files}
    model, joints, bodies = training_model()
    preview = mujoco.MjModel.from_xml_path(str(preview_xml.resolve()))
    data, visual = mujoco.MjData(model), mujoco.MjData(preview)
    names = reference["joint_names"].tolist()
    if len(names) != len(set(names)) or set(names) != set(joints):
        raise ValueError("Reference and training joint names differ")
    order = [names.index(n) for n in joints]
    joint_ids = np.array([model.joint(n).id for n in joints])
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    source_joints = np.array([preview.joint(n).id for n in joints])
    body_ids = np.array([model.body(n).id for n in bodies])
    time = reference["time"]
    dt = float(1 / reference["fps"])
    if len(time) < 3 or not np.isclose(dt, .02) or not np.allclose(np.diff(time), dt):
        raise ValueError("Reference must have uniform 50 Hz sampling")
    for key, width in (("joint_pos", len(joints)), ("root_pos", 3), ("root_quat_wxyz", 4)):
        if reference[key].shape != (len(time), width) or not np.isfinite(reference[key]).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}")
    if not np.allclose(np.linalg.norm(reference["root_quat_wxyz"], axis=-1), 1, atol=1e-6):
        raise ValueError("Root quaternions must be unit wxyz quaternions")
    poses = np.tile(model.qpos0, (len(time), 1))
    poses[:, :3], poses[:, 3:7] = reference["root_pos"], reference["root_quat_wxyz"]
    poses[:, qadr] = reference["joint_pos"][:, order]
    velocity = differentiate(model, poses, dt)
    feet = [model.geom(n).id for n in ("left_foot_collision", "right_foot_collision")]
    floor = model.geom("audit_floor").id
    arrays = {k: [] for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    contacts, com_margins, sole_min, axis_error, joint_origin_error = [], [], [], [], []
    sole_positions = []
    twist = np.empty(6)
    for i, pose in enumerate(poses):
        data.qpos[:], data.qvel[:] = pose, velocity[i]
        visual.qpos[:3], visual.qpos[3:7] = pose[:3], pose[3:7]
        visual.qpos[preview.jnt_qposadr[source_joints]] = poses[i, qadr]
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(preview, visual)
        axis_error.append(np.linalg.norm(data.xaxis[joint_ids] - visual.xaxis[source_joints], axis=1))
        joint_origin_error.append(np.linalg.norm(data.xanchor[joint_ids] - visual.xanchor[source_joints], axis=1))
        arrays["body_pos_w"].append(data.xpos[body_ids].copy())
        arrays["body_quat_w"].append(data.xquat[body_ids].copy())
        linear, angular = [], []
        for bid in body_ids:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), twist, 0)
            angular.append(twist[:3].copy())
            linear.append(twist[3:].copy())
        arrays["body_lin_vel_w"].append(linear)
        arrays["body_ang_vel_w"].append(angular)
        points = np.concatenate([geom_vertices(model, data, g) for g in feet])
        sole_positions.append(points)
        sole_min.append(float(points[:, 2].min()))
        hull = ConvexHull(points[:, :2])
        com = data.subtree_com[model.body("trunk_base").id]
        com_margins.append(float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max()))
        for c in data.contact:
            if c.dist >= -.0001:
                continue
            is_floor = floor in (c.geom1, c.geom2)
            other = c.geom2 if c.geom1 == floor else c.geom1
            contacts.append({"frame": i, "time": float(time[i]),
                             "kind": "foot_ground" if is_floor and other in feet else "nonfoot_ground" if is_floor else "self",
                             "pair": [geom_label(model, c.geom1), geom_label(model, c.geom2)],
                             "penetration_mm": float(-c.dist * 1000)})
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    # Keep quaternion signs continuous for downstream interpolation/differentiation.
    for i in range(1, len(time)):
        flip = (arrays["body_quat_w"][i] * arrays["body_quat_w"][i - 1]).sum(-1) < 0
        arrays["body_quat_w"][i, flip] *= -1
    np.savez_compressed(output / "motion.npz", **arrays, time=time, fps=reference["fps"],
                        quaternion_order=np.array("wxyz"), joint_names=np.array(joints), body_names=np.array(bodies),
                        joint_pos=poses[:, qadr], joint_vel=velocity[:, vadr],
                        root_pos=poses[:, :3], root_quat_wxyz=poses[:, 3:7],
                        qpos=np.c_[poses[:, :7], poses[:, qadr]], qvel=np.c_[velocity[:, :6], velocity[:, vadr]], loop_duration_s=np.array(float(time[-1])))
    nonfoot = [c for c in contacts if c["kind"] != "foot_ground"]
    margin = np.minimum(poses[:, qadr] - model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1] - poses[:, qadr])
    metrics = {
        "source": str(source.resolve()), "source_sha256": sha256(source),
        "training_xml": str(OPENCHICKEN_ALLCOLLISIONS_XML), "training_xml_sha256": sha256(OPENCHICKEN_ALLCOLLISIONS_XML),
        "preview_xml_sha256": sha256(preview_xml), "joint_names": joints, "body_names": bodies,
        "joint_mapping": "identity by name; no sign flips; body frames recomputed from training MJCF",
        "joint_axis_max_vector_error": float(np.max(axis_error)),
        "joint_anchor_max_error_mm": float(np.max(joint_origin_error) * 1000),
        "duration_s": float(time[-1]), "samples_with_endpoint": len(time), "loop_unique_samples": len(time) - 1,
        "robot_mass_kg": float(model.body_mass.sum()),
        "minimum_hard_joint_margin_deg": float(np.rad2deg(margin.min())),
        "minimum_sole_height_mm": float(min(sole_min) * 1000),
        "maximum_sole_vertex_drift_mm": float(np.linalg.norm(np.asarray(sole_positions) - sole_positions[0], axis=-1).max() * 1000),
        "minimum_com_support_hull_margin_mm": float(min(com_margins) * 1000),
        "support_margin_note": "Geometric projection only; not a dynamic balance proof or contact-force test.",
        "nonfoot_collision_frames": len(set(c["frame"] for c in nonfoot)),
        "maximum_nonfoot_penetration_mm": max((c["penetration_mm"] for c in nonfoot), default=0),
        "contacts": contacts,
        "mapping_pass": bool(np.max(axis_error) < 1e-6 and np.max(joint_origin_error) < .0001 and margin.min() >= 0),
        "collision_pass": not any(c["penetration_mm"] > .5 for c in nonfoot),
        "dynamics_checked": False,
    }
    metrics["training_reference_ready"] = False  # Requires the independent physics gate too.
    (output / "alignment.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "fps": 1 / dt, "quaternion_order": "wxyz",
        "clips": [{"name": "openchicken_side_lean_v12", "file": "motion.npz", "frames": len(time)}],
        "source_sha256": sha256(source), "motion_sha256": sha256(output / "motion.npz"),
        "validation": "pending_physics" if metrics["collision_pass"] else "failed_collision",
    }, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "contacts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preview-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.preview_xml, args.output)
