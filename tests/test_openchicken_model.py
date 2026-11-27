"""OpenChicken resources, policy indexing and independent beak control."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg

from mjlab_openchicken.robot import openchicken_constants as robot
from mjlab_openchicken.tasks import mdp
from mjlab_openchicken.tasks.openchicken_roller_standup_env_cfg import _LEG_JOINTS


MODEL_DIR = Path(robot.OPENCHICKEN_WALK_XML).parent


@pytest.mark.parametrize("task", [name for name in list_tasks() if "OpenChicken" in name])
def test_every_task_selects_exactly_fourteen_policy_actuators(task):
    for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        entity = Entity(deepcopy(cfg.scene.entities["robot"]))
        _, names = entity.find_joints_by_actuator_names(cfg.actions["joint_pos"].actuator_names)
        assert len(names) == 14
        assert names == entity.actuators[0].target_names
        assert "passive_beak" not in names


@pytest.mark.parametrize("path", sorted(MODEL_DIR.glob("robot_*.xml")) +
                         sorted((MODEL_DIR / "scenes").glob("scene*.xml")), ids=lambda p: p.name)
def test_models_and_scenes_resolve_all_assets(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nu == 15
    assert model.joint("passive_beak").id >= 0
    assert model.actuator("beak_open").id >= 0
    # Bearing meshes remain visible but never participate in contacts.
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = model.mesh(int(model.geom_dataid[g])).name
            if mesh_name.startswith("seeed_bearing__"):
                assert model.geom_contype[g] == model.geom_conaffinity[g] == 0
    for body in ("hip_l", "hip_l_2"):
        stem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, body + "_stem_collision")
        if stem >= 0:
            assert model.geom_type[stem] == mujoco.mjtGeom.mjGEOM_MESH
            assert model.geom(body + "_end_collision").type[0] == mujoco.mjtGeom.mjGEOM_MESH
    for k in range(model.nkey):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, k)
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all()
        assert data.ctrl[model.actuator("beak_open").id] == 0


@pytest.mark.parametrize("cfg", [robot.OPENCHICKEN_WALK_ROBOT_CFG,
                                 robot.OPENCHICKEN_STANDUP_ROBOT_CFG,
                                 robot.OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG,
                                 robot.OPENCHICKEN_WALK_ROLLERS_ROBOT_CFG,
                                 robot.OPENCHICKEN_BACKLASH_ROBOT_CFG,
                                 robot.OPENCHICKEN_ROLLERS_BACKLASH_ROBOT_CFG])
def test_beak_is_separate_from_policy_and_leg_reward_indices(cfg):
    entity = Entity(deepcopy(cfg))
    model = entity.spec.compile()
    servos, beak = entity.actuators
    assert len(servos.target_names) == 14
    assert beak.target_names == ["passive_beak"]
    assert model.actuator("beak_open").gainprm[0] == .16
    ids = mdp._servo_joint_ids(SimpleNamespace(), entity)
    assert [entity.joint_names[i] for i in ids] == servos.target_names
    assert [entity.joint_names[ids[i]] for i in _LEG_JOINTS] == [
        f"{side}_{joint}" for side in ("left", "right")
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    ]


def test_roller_pose_reward_ignores_beak_and_wheels():
    entity = Entity(deepcopy(robot.OPENCHICKEN_WALK_ROLLERS_ROBOT_CFG))
    names = entity.joint_names
    data = SimpleNamespace(
        joint_pos=torch.zeros(2, len(names)),
        default_joint_pos=torch.zeros(2, len(names)),
        root_link_pos_w=torch.tensor([[0., 0., .12]] * 2),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2),
    )
    asset = SimpleNamespace(data=data, find_joints=entity.find_joints)
    scene = {"robot": asset}
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    env = SimpleNamespace(scene=Scene(scene))
    kwargs = dict(target_height=.12, height_std=.02, upright_std=.2,
                  pose_std=.2, joint_indices=_LEG_JOINTS)
    baseline = mdp.standing_composite_score(env, **kwargs)
    data.joint_pos[:, [i for i, n in enumerate(names) if n.startswith("passive_")]] = .2
    torch.testing.assert_close(mdp.standing_composite_score(env, **kwargs), baseline)
    data.joint_pos[:, names.index("right_hip_pitch")] = .2
    assert (mdp.standing_composite_score(env, **kwargs) < baseline).all()
