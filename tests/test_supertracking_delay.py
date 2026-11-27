"""Transport timing, delayed credit assignment, and real BAM integration."""
import itertools

import pytest
import torch

from mjlab.utils.buffers import DelayBuffer
from mjlab_openchicken.supertracking.delay import command_target, delayed_targets
from mjlab_openchicken.supertracking.models import WorldModel, Critic


def test_all_lag_sequences_match_mjlab_including_empty_and_partial_history():
    patterns = torch.tensor(list(itertools.product(range(3, 7), repeat=4)))
    n = len(patterns)
    for warmup in (0, 1, 3, 7, 12):
        buffer = DelayBuffer(3, 6, n)
        for i in range(warmup):
            buffer.append(torch.full((n, 14), float(i)))
        queue = buffer._buffer.buffer.clone() if warmup else torch.zeros(n, 7, 14)
        length = buffer._buffer.current_length.clone()
        target = torch.arange(14).float().expand(n, -1) + 20
        expected, next_queue, next_length = delayed_targets(queue, length, target, patterns)
        actual = []
        for j in range(4):
            buffer.append(target)
            buffer._update_lags = lambda j=j: buffer.set_lags(patterns[:, j])
            actual.append(buffer.compute())
        torch.testing.assert_close(expected, torch.stack(actual, 1), atol=0, rtol=0)
        torch.testing.assert_close(next_queue, buffer._buffer.buffer, atol=0, rtol=0)
        torch.testing.assert_close(next_length, buffer._buffer.current_length)


def test_delayed_credit_reaches_earlier_action_without_current_action_shortcut():
    queue, length = torch.zeros(2, 7, 14), torch.full((2,), 7)
    first = torch.randn(2, 14, requires_grad=True)
    second = torch.randn(2, 14, requires_grad=True)
    lags = torch.full((2, 4), 6)
    applied1, queue, length = delayed_targets(queue, length, first, lags)
    applied2, _, _ = delayed_targets(queue, length, second, lags)
    assert not applied1.count_nonzero()
    torch.testing.assert_close(applied2[:, :2], torch.zeros(2, 2, 14))
    torch.testing.assert_close(applied2[:, 2:], first[:, None].expand(-1, 2, -1))
    applied2.sum().backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, 2))
    torch.testing.assert_close(second.grad, torch.zeros_like(second))

    world = WorldModel(command_delay=True)
    torch.nn.init.normal_(world.net[-1].weight, std=.01)
    state = torch.randn(2, 83) * .01
    state[:, 3] = 1
    alternative = state.clone()
    alternative[:, 55:83] += 10
    result1 = world(state, first.detach(), applied1.detach())
    result2 = world(alternative, second.detach(), applied1.detach())
    torch.testing.assert_close(result1[:, :55], result2[:, :55], atol=0, rtol=0)
    torch.testing.assert_close(result2[:, 55:69], second.detach())
    with pytest.raises(ValueError, match='four ordered BAM targets'):
        world(state, first)


def test_critic_observes_pending_commands_but_not_future_lags():
    critic = Critic(command_delay=True)
    state, queue = torch.zeros(2, 83), torch.randn(2, 7, 14, requires_grad=True)
    length = torch.full((2,), 7)
    future = torch.zeros(2, 10, 41)
    future[..., 3] = 1
    critic(state, future, queue, length).sum().backward()
    assert torch.isfinite(queue.grad).all() and queue.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Replay transfer requires CUDA')
def test_cpu_replay_preserves_gpu_samples_and_episode_masks():
    from mjlab_openchicken.supertracking.runner import Replay

    gpu, cpu = Replay(12, 16, 'cuda:0'), Replay(12, 16, 'cuda:0', 'cpu')
    for step in range(18):
        transition = {'state': torch.randn(16, 83, device='cuda:0'),
                      'command_queue': torch.randn(16, 7, 14, device='cuda:0'),
                      'command_lags': torch.randint(3, 7, (16, 4), device='cuda:0'),
                      'episode': torch.full((16,), step // 3, device='cuda:0')}
        gpu.add(transition)
        cpu.add(transition)
    assert all(v.device.type == 'cpu' for v in cpu.data.values())
    for validation in (False, True):
        torch.manual_seed(421)
        expected = gpu.sample(64, 8, validation)
        torch.manual_seed(421)
        actual = cpu.sample(64, 8, validation)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='MuJoCo Warp integration requires CUDA')
def test_real_bam_targets_reset_isolation_and_autoregressive_reconstruction():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_openchicken.tasks.openchicken_supertracking_env_cfg import make_openchicken_supertracking_env_cfg, ROBUST_MOTION
    from mjlab_openchicken.supertracking.env import SupertrackingEnv

    cfg = make_openchicken_supertracking_env_cfg(robust=True, push_test=True)
    cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        tracking = SupertrackingEnv(env, ROBUST_MOTION)
        queue, length = tracking.command_delay.snapshot()
        assert (length == 7).all()
        torch.testing.assert_close(queue, tracking.robot.data.joint_pos_target[:, tracking.servo_ids, None].transpose(1, 2).expand(-1, 7, -1))
        for step in range(32):
            action = tracking.motion.state[tracking.frame, 7:21] - tracking.default
            # Repeated command reversals exercise the queue near backlash changes.
            action = action + (.06 if step % 2 else -.06)
            tr, _, _ = tracking.step(action, reset=False)
            torch.testing.assert_close(queue, tr['command_queue'], atol=0, rtol=0)
            targets, queue, length = delayed_targets(queue, length, command_target(tr['state'], tr['action'], tracking.default), tr['command_lags'])
            torch.testing.assert_close(targets, tr['applied_targets'], atol=1e-7, rtol=0)
            actual_queue, actual_length = tracking.command_delay.snapshot()
            torch.testing.assert_close(queue, actual_queue, atol=1e-7, rtol=0)
            torch.testing.assert_close(length, actual_length)
            if step in (3, 15):
                actuator = tracking.command_delay.actuator
                torque = actuator._prev_motor_torque.clone()
                counts = tracking.command_delay.buffer._step_count.clone()
                tracking.reset(ids=torch.tensor([0, 2, 7], device='cuda:0'), frame=0)
                unchanged = torch.ones(64, dtype=torch.bool, device='cuda:0')
                unchanged[[0, 2, 7]] = False
                after, after_length = tracking.command_delay.snapshot()
                torch.testing.assert_close(after[unchanged], queue[unchanged], atol=0, rtol=0)
                torch.testing.assert_close(actuator._prev_motor_torque[unchanged], torque[unchanged], atol=0, rtol=0)
                torch.testing.assert_close(tracking.command_delay.buffer._step_count[unchanged], counts[unchanged])
                queue, length = after, after_length
        manager = env.event_manager
        index = manager.active_terms['interval'].index('push_robot')
        manager._interval_term_time_left[index][:] = 0
        tr, _, _ = tracking.step(torch.zeros(64, 14, device='cuda:0'), reset=False)
        torch.testing.assert_close(tr['next_state'][:, 21:24], env.sim.data.qvel[:, :3], atol=1e-6, rtol=1e-6)
    finally:
        env.close()
