"""BAM command transport for real transitions and differentiable imagination.

The adapter is private to SupertrackingEnv; native actuators are unchanged.
Queues are chronological, absolute joint targets, at the 5 ms physics rate.
"""
from contextlib import contextmanager

import torch


def command_target(state, action, default):
    """Match mjlab JointPositionAction: HOME + action - encoder offset."""
    target = action + default
    return target - state[..., 111:125] if state.shape[-1] > 83 else target


def delayed_targets(queue, length, target, lags):
    """Append the held target before each lag lookup, exactly as mjlab does."""
    applied = []
    for lag in lags.unbind(-1):
        queue = torch.cat((queue[..., 1:, :], target[..., None, :]), -2)
        first = length == 0
        queue = torch.where(first[..., None, None], target[..., None, :], queue)
        length = (length + 1).clamp_max(queue.shape[-2])
        valid = torch.minimum(lag, length - 1).clamp_min(0)
        index = (queue.shape[-2] - 1 - valid)[..., None, None]
        applied.append(queue.gather(-2, index.expand(*index.shape[:-1], target.shape[-1])).squeeze(-2))
    return torch.stack(applied, -2), queue, length


class BamCommandDelay:
    """Observe actual delay choices without sampling or evaluating BAM twice."""

    def __init__(self, robot, decimation, physics_dt):
        if len(robot.actuators) != 1 or decimation != 4 or abs(physics_dt - .005) > 1e-9:
            raise ValueError('Expected one 14-servo BAM group, 4 x 5 ms substeps')
        self.robot, self.actuator = robot, robot.actuators[0]
        self.buffer = self.actuator._delay_buffer
        cfg = self.actuator.cfg
        if (cfg.delay_min_lag, cfg.delay_max_lag, cfg.delay_update_period, cfg.delay_hold_prob) != (3, 6, 0, 0):
            raise ValueError('Command transport expects independent uniform BAM lags 3..6')
        if self.buffer is None or not self.buffer.per_env or len(self.actuator.target_ids) != 14:
            raise ValueError('Expected one shared lag per environment for 14 position targets')
        self._apply_delay = self.actuator.apply_delay
        self.actuator.apply_delay = self._record_delay
        self.recording = False
        self.lags, self.targets = [], []

    def _record_delay(self, command):
        delayed = self._apply_delay(command)
        if self.recording:
            self.lags.append(self.buffer.current_lags.clone())
            self.targets.append(delayed.position_target.clone())
        return delayed

    def snapshot(self):
        return self.buffer._buffer.buffer.clone(), self.buffer._buffer.current_length.clone()

    @contextmanager
    def record(self):
        self.lags, self.targets = [], []
        self.recording = True
        try:
            yield
        finally:
            self.recording = False

    def transition(self):
        if len(self.lags) != 4:
            raise RuntimeError(f'Expected exactly four BAM updates, got {len(self.lags)}')
        return torch.stack(self.lags, 1), torch.stack(self.targets, 1)

    @contextmanager
    def reset_without_actuation(self):
        # mjlab.reset calls scene.write_data_to_sim without advancing physics.
        # Suppress this instance's BAM evaluation, including torque/sag updates.
        write = self.robot.write_data_to_sim
        self.robot.write_data_to_sim = lambda: None
        try:
            yield
        finally:
            self.robot.write_data_to_sim = write

    def seed(self, ids, targets):
        # Fill reset rows with the held reset posture. Do not append: that would
        # rotate the shared circular pointer and advance every other environment.
        history = self.buffer._buffer
        if history._buffer is None:
            history._buffer = targets.new_zeros((7, self.buffer.batch_size, 14))
            history._pointer = 0
        history._buffer[:, ids] = targets[None]
        history._num_pushes[ids] = 7

    def sample(self, batch):
        return torch.randint(3, 7, (batch, 4), device=self.buffer.device)
