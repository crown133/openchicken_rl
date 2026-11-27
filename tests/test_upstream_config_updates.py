"""Training defaults must stay independent of viewers and match the local robot."""
from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.entity import Entity

from mjlab_openchicken.robot.openchicken_constants import OPENCHICKEN_GROUNDCONTACT_ROLLERS_XML
from mjlab_openchicken.tasks import mdp
from mjlab_openchicken.tasks import openchicken_velstand_env_cfg as velstand
from mjlab_openchicken.tasks.openchicken_velocity_env_cfg import (
    OPENCHICKEN_ROUGH_TERRAINS_CFG, make_openchicken_velocity_env_cfg,
)
from mjlab_openchicken.tasks.openchicken_standup_env_cfg import make_openchicken_standup_env_cfg
from mjlab_openchicken.tasks.openchicken_sitstand_env_cfg import make_openchicken_sitstand_env_cfg
from mjlab_openchicken.tasks.openchicken_ground_pick_env_cfg import make_openchicken_ground_pick_env_cfg


@pytest.mark.parametrize("factory", [make_openchicken_velocity_env_cfg,
    velstand.make_openchicken_velstand_env_cfg, make_openchicken_standup_env_cfg,
    make_openchicken_sitstand_env_cfg, make_openchicken_ground_pick_env_cfg])
@pytest.mark.parametrize("play_first", [False, True])
def test_terrain_factories_are_independent(factory, play_first):
    before = deepcopy(OPENCHICKEN_ROUGH_TERRAINS_CFG)
    configs = {play: factory(rough=True, play=play) for play in (play_first, not play_first)}
    train = configs[False].scene.terrain.terrain_generator
    play = configs[True].scene.terrain.terrain_generator
    assert (train.num_rows, train.num_cols) == (10, 20)
    assert (play.num_rows, play.num_cols) == (5, 5)
    assert train is not play and train is not OPENCHICKEN_ROUGH_TERRAINS_CFG
    assert train.curriculum == before.curriculum  # Isolation does not change sampling.
    key = "pyramid_stairs"
    play.sub_terrains[key].proportion = .99
    assert train.sub_terrains[key].proportion == before.sub_terrains[key].proportion
    assert OPENCHICKEN_ROUGH_TERRAINS_CFG == before


@pytest.mark.parametrize("task", [name for name in list_tasks()
    if "OpenChicken" in name and "Rough" in name])
def test_registered_rough_training_grid_survives_play_registration(task):
    train = load_env_cfg(task).scene.terrain.terrain_generator
    play = load_env_cfg(task, play=True).scene.terrain.terrain_generator
    assert (train.num_rows, train.num_cols) == (10, 20)
    assert (play.num_rows, play.num_cols) == (5, 5)


def test_velstand_default_keeps_yaw_calibration_without_v8_recovery_changes():
    cfg = velstand.make_openchicken_velstand_env_cfg()
    bc = velstand.OpenChickenVelStandRlCfg.algorithm.bc_cfg
    assert velstand.ENABLE_ANCHOR_YAW_REMAP
    assert not velstand.ENABLE_JOINT_MASKED_TEACHER
    assert not velstand.ENABLE_WIDE_SPAWNS
    assert bc["anchor_yaw_remap"] == velstand.ANCHOR_YAW_REMAP
    assert not bc.get("expert_mask_cols")
    assert cfg.rewards["track_yaw_fine"].weight > 0
    spawn = cfg.events["random_prone_init"].params
    assert all(spawn.get(key, 0.) == 0. for key in ("so3_prob", "sit_prob", "joint_random_prob"))


@pytest.fixture(scope="module")
def roller_stance_height():
    scene = OPENCHICKEN_GROUNDCONTACT_ROLLERS_XML.parent / "scenes/scene_rollers.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    data.qpos[2] = .14
    # A short settling window: passive wheels do not hold a static balance.
    heights = []
    while data.time < .3:
        mujoco.mj_step(model, data)
        if data.time >= .15:
            quat = data.xquat[model.body("trunk_base").id]
            tilt = np.arccos(np.clip(1 - 2 * (quat[1]**2 + quat[2]**2), -1, 1))
            assert tilt < np.deg2rad(5), "A fallen pose is not a standing-height measurement"
            heights.append(data.xpos[model.body("trunk_base").id, 2])
    return float(np.mean(heights))


@pytest.mark.parametrize("task", [name for name in list_tasks()
    if "OpenChicken" in name and ("Rollers" in name or "Swizzle" in name)])
def test_roller_reward_covers_stance_and_spawn_without_paying_for_deep_crouch(task, roller_stance_height):
    cfg = load_env_cfg(task)
    term = cfg.rewards["com_height_target"]
    low, high = term.params["target_height_min"], term.params["target_height_max"]
    spawn_low, spawn_high = cfg.events["reset_base"].params["pose_range"]["z"]
    assert low <= spawn_low <= spawn_high <= high
    assert low < roller_stance_height < high
    assert .005 < roller_stance_height - low < .02
    heights = torch.tensor([roller_stance_height, spawn_low, spawn_high, roller_stance_height - .03])
    position = torch.zeros(4, 3)
    position[:, 2] = heights
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(4, 3))
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=position))))
    rewards = mdp.com_height_target(env, **term.params) * term.weight
    assert (rewards[:3] > 0).all()
    assert rewards[3] <= 0


@pytest.mark.parametrize("task", ["Mjlab-VelStand-Flat-OpenChicken",
    "Mjlab-VelStand-Flat-Backlash-OpenChicken", "Mjlab-Velocity-Flat-OpenChicken-Rollers"])
def test_stall_reward_matches_policy_joint_velocities_and_excludes_beak(task):
    entity = Entity(deepcopy(load_env_cfg(task).scene.entities['robot']))
    torque = torch.zeros(2, entity.num_actuators)
    velocity = torch.zeros(2, len(entity.joint_names))
    leg_actuator = entity.actuator_names.index('left_hip_pitch')
    leg_joint = entity.joint_names.index('left_hip_pitch')
    torque[:, leg_actuator] = .9
    torque[:, entity.actuator_names.index('beak_open')] = 9.
    velocity[1, leg_joint] = 3.
    asset = SimpleNamespace(find_joints=entity.find_joints, find_actuators=entity.find_actuators,
        data=SimpleNamespace(actuator_force=torque, joint_vel=velocity))
    env = SimpleNamespace(scene={'robot': asset})
    assert mdp.servo_stall_penalty(env).tolist() == [1., 0.]
