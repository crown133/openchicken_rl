"""Model-based tracking contracts and physics-coordinate regression tests."""
from copy import deepcopy
import numpy as np
import mujoco
import torch
import pytest
from mjlab.entity import Entity
from mjlab_openchicken.robot.openchicken_constants import OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG
from mjlab_openchicken.tasks.openchicken_supertracking_env_cfg import make_openchicken_supertracking_env_cfg
from mjlab_openchicken.tasks import mdp
from mjlab_openchicken.supertracking.models import Kinematics, Actor, WorldModel, Critic, goal_features
from mjlab_openchicken.supertracking.runner import Replay, frozen
from mjlab_openchicken.supertracking.delay import delayed_targets


def fixture():
    entity = Entity(deepcopy(OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG))
    model = entity.spec.compile()
    bodies = [model.body(n).id for n in entity.body_names if n not in ("comb", "lower_beak")]
    joints = [model.joint(n).id for n in entity.joint_names if not n.startswith("passive_")]
    return model, bodies, joints


def test_differentiable_kinematics_matches_mujoco_and_has_gradients():
    model, bodies, joints = fixture()
    kinematics = Kinematics(model, bodies, joints)
    poses = np.tile(model.qpos0, (5, 1))
    rng = np.random.default_rng(42)
    poses[:, model.jnt_qposadr[joints]] = rng.uniform(model.jnt_range[joints, 0], model.jnt_range[joints, 1], (5, 14))
    poses[:, 3:7] = rng.normal(size=(5, 4))
    poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=-1, keepdims=True)
    state = torch.tensor(np.c_[poses[:, :7], poses[:, model.jnt_qposadr[joints]], np.zeros((5, 62))], dtype=torch.float32, requires_grad=True)
    pos, quat = kinematics(state)
    data = mujoco.MjData(model)
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(pos[i].detach(), data.xpos[bodies], atol=2e-7)
        np.testing.assert_allclose(abs((quat[i].detach().numpy() * data.xquat[bodies]).sum(-1)), 1, atol=3e-7)
    pos.square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad[:, 7:21].abs().sum() > 0


def test_replay_masks_episode_boundaries_and_holds_out_envs():
    replay = Replay(12, 16, 'cpu')
    for i in range(18):
        replay.add({'episode': torch.full((16,), i // 3), 'state': torch.arange(16)[:, None].float(),
                    'frame': torch.full((16,), i)})
    train = replay.sample(128, 8)
    valid = replay.sample(128, 8, validation=True)
    assert train['state'].max() < 14 and valid['state'].min() >= 14
    assert (train['frame'][:, 1:] - train['frame'][:, :-1] == 1).all()
    assert torch.equal(train['mask'], train['episode'] == train['episode'][:, :1])
    assert (train['mask'].sum(-1) <= 3).all()


def test_reward_gradients_reach_actor_through_frozen_world():
    model, bodies, joints = fixture()
    actor = Actor(torch.zeros(14), torch.tensor(model.jnt_range[joints], dtype=torch.float32))
    world, critic = WorldModel(command_delay=True), Critic()
    # Trained dynamics must depend on actions; initialize a small known Jacobian.
    torch.nn.init.normal_(world.net[-1].weight, std=.001)
    state = torch.zeros(4, 83)
    state[:, 2], state[:, 3] = .1, 1
    history = mdp.supertracking_observation(state, torch.zeros(14))[:, None].repeat(1, 10, 1)
    future = state[:, None, :41].repeat(1, 10, 1)
    reference = state[:, :41].clone()
    reference[:, 1] = .01
    fk = Kinematics(model, bodies, joints)
    ref_pos, ref_quat = fk(reference)
    queue, length = torch.zeros(4, 7, 14), torch.full((4,), 7)
    with frozen(world, critic):
        for _ in range(3):
            action = actor(history, future)
            targets, queue, length = delayed_targets(queue, length, action, torch.full((4, 4), 6))
            state = world(state, action, targets)
            obs = mdp.supertracking_observation(state, actor.default_q)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
        pos, quat = fk(state)
        reward, _ = mdp.supertracking_reward(state, reference, pos, quat, ref_pos, ref_quat, action, torch.zeros_like(action))
        (-reward.mean()).backward()
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in world.parameters())
    assert all(p.requires_grad for p in world.parameters())
    assert goal_features(future).shape == (4, 430)


def test_config_retains_bam_collisions_and_original_actor_layout():
    cfg = make_openchicken_supertracking_env_cfg()
    assert 'expand_bam_friction_fields' in cfg.events
    actuator = cfg.scene.entities['robot'].articulation.actuators[0]
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (3, 6)
    assert cfg.auto_reset is False
    assert list(cfg.observations['actor'].terms) == list(
        __import__('mjlab_openchicken.tasks.openchicken_velocity_env_cfg', fromlist=['make_openchicken_velocity_env_cfg'])
        .make_openchicken_velocity_env_cfg().observations['actor'].terms)
    assert not cfg.rewards and not cfg.terminations


def test_failure_and_reward_are_consistent_at_reference():
    model, bodies, joints = fixture()
    state = torch.zeros(2, 83)
    state[:, 2], state[:, 3] = .1, 1
    pos, quat = Kinematics(model, bodies, joints)(state)
    action = torch.zeros(2, 14)
    reward, _ = mdp.supertracking_reward(state, state[:, :41], pos, quat, pos, quat, action, action)
    torch.testing.assert_close(reward, torch.ones(2))
    assert not mdp.supertracking_failure(state, state[:, :41]).any()
    fallen = state.clone()
    fallen[:, 2] = .04
    assert mdp.supertracking_failure(fallen, state[:, :41]).all()
    assert (mdp.supertracking_failure(fallen, state[:, :41], soft=True) > .95).all()


def test_backlash_encoder_fk_matches_physical_links():
    from mjlab_openchicken.robot.openchicken_constants import OPENCHICKEN_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
    entity = Entity(deepcopy(OPENCHICKEN_ALLCOLLISIONS_BACKLASH_ROBOT_CFG))
    model = entity.spec.compile()
    data = mujoco.MjData(model)
    nominal, bodies, joints = fixture()
    fk = Kinematics(nominal, bodies, joints)
    names = [nominal.joint(j).name for j in joints]
    target_bodies = [model.body(nominal.body(b).name).id for b in bodies]
    rng = np.random.default_rng(27)
    for _ in range(6):
        state = torch.zeros(1, 111)
        state[0, :7] = torch.tensor(model.qpos0[:7])
        for k, name in enumerate(names):
            j, p = model.joint(name).id, model.joint(f'passive_{name}_backlash').id
            motor = rng.uniform(*model.jnt_range[j])
            play = rng.uniform(-np.deg2rad(1), np.deg2rad(1))
            data.qpos[model.jnt_qposadr[j]] = motor
            data.qpos[model.jnt_qposadr[p]] = play
            state[0, 7+k] = motor + play
        data.qpos[:7] = state[0, :7]
        mujoco.mj_forward(model, data)
        pos, quat = fk(state)
        np.testing.assert_allclose(pos[0], data.xpos[target_bodies], atol=3e-7)
        np.testing.assert_allclose(abs((quat[0].numpy() * data.xquat[target_bodies]).sum(-1)), 1, atol=4e-7)


def test_sensor_bias_and_delays_use_raw_previous_sample():
    state = torch.zeros(2, 140)
    state[:, 3] = 1
    state[:, 125] = 1
    state[:, 111:125] = .01
    state[:, 129] = 1  # delayed gyro, current gravity
    state[:, 24:27] = 2
    state[:, 27:41] = 3
    previous = torch.full((2, 48), 7.)
    obs, raw = mdp.supertracking_sensor_observation(state, torch.zeros(14), previous)
    torch.testing.assert_close(obs[:, :3], torch.full((2, 3), 7.))
    torch.testing.assert_close(obs[:, 6:20], torch.full((2, 14), .01))
    torch.testing.assert_close(obs[:, 20:34], torch.full((2, 14), 7.))
    torch.testing.assert_close(raw[:, 20:34], torch.full((2, 14), 3.))
    # Selective environment reset must not overwrite this raw predecessor.
    obs2, _ = mdp.supertracking_sensor_observation(state, torch.zeros(14), raw)
    torch.testing.assert_close(obs2[:, 20:34], torch.full((2, 14), 3.))


def test_robust_world_keeps_context_and_models_passive_motion():
    world = WorldModel(150, backlash=True)
    state = torch.randn(3, 150) * .01
    state[:, 3] = 1
    state[:, 125] = 1
    action = torch.zeros(3, 14)
    predicted = world(state, action)
    assert predicted.shape == state.shape
    torch.testing.assert_close(predicted[:, 111:], state[:, 111:])
    torch.testing.assert_close(predicted[:, 83:97], state[:, 83:97] + .02 * state[:, 97:111])
    predicted.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in world.parameters() if p.grad is not None)


def test_robust_config_uses_correct_dr_scopes_without_polluting_nominal():
    cfg = make_openchicken_supertracking_env_cfg(robust=True)
    assert cfg.episode_length_s == 12
    assert cfg.scene.entities['robot'].articulation.actuators[0].vin_range == (6.5, 8.2)
    assert {'foot_friction', 'encoder_bias', 'randomize_com', 'randomize_head_com',
            'randomize_mass_inertia', 'randomize_joint_friction', 'randomize_armature',
            'expand_bam_friction_fields'} <= cfg.events.keys()
    assert 'push_robot' not in cfg.events
    push_test = make_openchicken_supertracking_env_cfg(robust=True, push_test=True)
    assert push_test.events['push_robot'].params['velocity_range'] == {'x': (-.30, .30), 'y': (-.30, .30)}
    assert 'bearing_roll' not in cfg.events['randomize_head_com'].params['asset_cfg'].body_names
    assert cfg.events['randomize_armature'].params['asset_cfg'].joint_names == (r'^(?!passive_).*',)
    nominal = make_openchicken_supertracking_env_cfg()
    assert nominal.scene.entities['robot'].articulation.actuators[0].vin_range == (7.4, 7.4)
    assert 'push_robot' not in nominal.events


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("play", [False, True])
def test_supertracking_uses_full_collision_model_with_independent_beak(robust, play):
    cfg = make_openchicken_supertracking_env_cfg(robust=robust, play=play)
    robot = Entity(deepcopy(cfg.scene.entities["robot"]))
    model = robot.spec.compile()
    active = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    collision_bodies = {model.body(int(model.geom_bodyid[i])).name for i in active}
    assert len(active) == 42
    assert {"upper_leg_left", "upper_leg_right", "lower_beak"} <= collision_bodies
    assert not {"comb", "neck", "neck_pitch"} & collision_bodies
    assert sum(n.endswith("_backlash") for n in robot.joint_names) == (14 if robust else 0)
    _, policy_names = robot.find_joints_by_actuator_names(cfg.actions["joint_pos"].actuator_names)
    assert len(policy_names) == 14 and "passive_beak" not in policy_names
    assert robot.actuators[1].target_names == ["passive_beak"]
