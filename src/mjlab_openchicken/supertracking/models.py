"""OpenChicken dynamics, differentiable MJCF kinematics and motion-conditioned actor.

Algorithm reference: Breezewrf/supermimic_mjlab at a7cee74636a4c596cd8710f8c2e51f82d78bb150.
This implementation uses OpenChicken BAM state and an independent policy schema.
"""
import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from mjlab_openchicken.tasks.mdp import supertracking_observation


def normalize_quat(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def rotation_quat(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), .5 * torch.sinc(angle / (2 * torch.pi)) * v), -1)


def mlp(widths, layer_norm=False):
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(b))
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(0.))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        count = len(x)
        total = self.count + count
        delta = x.mean(0) - self.mean
        new_mean = self.mean + delta * count / total
        self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * count
                        + delta.square() * self.count * count / total) / total)
        self.mean.copy_(new_mean)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / self.var.clamp_min(1e-4).sqrt()).clamp(-10, 10)


class Kinematics(nn.Module):
    def __init__(self, model, body_ids, joint_ids):
        super().__init__()
        bodies = list(body_ids)
        joints = list(joint_ids)
        self.parents = [-1] + [bodies.index(int(model.body_parentid[b])) for b in bodies[1:]]
        self.joints = [-1] + [joints.index(int(model.body_jntadr[b])) for b in bodies[1:]]
        for b in bodies[1:]:
            if model.body_jntnum[b] != 1:
                raise ValueError("Supertracking expects one hinge per non-root body")
        for name, value in (("offset", model.body_pos[bodies]), ("rotation", model.body_quat[bodies]),
                            ("axis", model.jnt_axis[joints]), ("anchor", model.jnt_pos[joints])):
            self.register_buffer(name, torch.tensor(value.copy(), dtype=torch.float32))

    def forward(self, state):
        positions, rotations = [state[..., :3]], [state[..., 3:7]]
        for i in range(1, len(self.parents)):
            j, parent = self.joints[i], self.parents[i]
            turn = rotation_quat(state[..., 7 + j:8 + j] * self.axis[j])
            local_q = quat_mul(self.rotation[i].expand_as(turn), turn)
            pivot = self.anchor[j].expand_as(state[..., :3])
            offset = self.offset[i] + quat_apply(self.rotation[i].expand_as(turn), pivot - quat_apply(turn, pivot))
            positions.append(positions[parent] + quat_apply(rotations[parent], offset))
            rotations.append(quat_mul(rotations[parent], local_q))
        return torch.stack(positions, -2), torch.stack(rotations, -2)


def goal_features(reference):
    """Ten future frames, 43 features each; retain absolute stance height."""
    q = reference[..., 3:7]
    ex, ey = torch.zeros_like(reference[..., :3]), torch.zeros_like(reference[..., :3])
    ex[..., 0], ey[..., 1] = 1, 1
    return torch.cat((reference[..., :3] / .1, quat_apply(q, ex), quat_apply(q, ey),
                      reference[..., 21:24], reference[..., 24:27] / 3,
                      reference[..., 7:21], reference[..., 27:41] / 5), -1).flatten(-2)


class Actor(nn.Module):
    def __init__(self, default_q, limits):
        super().__init__()
        self.register_buffer("default_q", default_q.clone())
        self.register_buffer("lower", limits[:, 0] + .01 - default_q)
        self.register_buffer("upper", limits[:, 1] - .01 - default_q)
        self.obs_norm = RunningNorm(48)
        self.goal_norm = RunningNorm(430)
        self.encoder = mlp((430, 256, 128, 18))
        self.decoder = mlp((512, 256, 128, 14))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, future):
        features = goal_features(future)
        # The 32D latent contains 14 reference commands + 18 learned features.
        # This supplies a useful initial reference controller and can be cached.
        return torch.cat((future[..., 0, 7:21] - self.default_q,
                          self.encoder(self.goal_norm(features)).tanh()), -1)

    def decode(self, history, latent):
        x = torch.cat((self.obs_norm(history).flatten(-2), latent), -1)
        command = latent[..., :14] + .35 * self.decoder(x).tanh()
        return torch.maximum(torch.minimum(command, self.upper), self.lower)

    def forward(self, history, future):
        return self.decode(history, self.encode(future))


class WorldModel(nn.Module):
    def __init__(self, state_dim=83, backlash=False, command_delay=False):
        super().__init__()
        self.backlash = backlash
        self.command_delay = command_delay
        width = state_dim + (28 if command_delay else 14)
        self.norm = RunningNorm(width)
        self.net = mlp((width, 512, 512, 256, 82 if backlash else 54), layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("delta_scale", torch.tensor(
            [.01]*3 + [.10]*3 + [.10]*14 + [.30]*3 + [2.]*3 + [5.]*14 + [.20]*14
            + ([.02]*14 + [2.]*14 if backlash else [])))

    def features(self, state, action, applied_targets=None):
        if self.command_delay:
            if applied_targets is None or applied_targets.shape[-2:] != (4, 14):
                raise ValueError('Delay-conditioned dynamics need four ordered BAM targets')
            # Requested commands only update bookkeeping below. Physics sees
            # the targets that actually reached BAM, with no direct shortcut.
            return torch.cat((state[..., :55], state[..., 83:], applied_targets.flatten(-2)), -1)
        return torch.cat((state, action), -1)

    def forward(self, state, action, applied_targets=None):
        delta = self.net(self.norm(self.features(state, action, applied_targets))) * self.delta_scale
        dp, dr, dq, dv, dw, dqd, torque = delta[..., :54].split((3, 3, 14, 3, 3, 14, 14), -1)
        q = state[..., 3:7]
        # A velocity-residual baseline reduces the initial multi-step fit burden.
        next_q = normalize_quat(quat_mul(q, rotation_quat(quat_apply_inverse(q, state[..., 24:27]) * .02 + dr)))
        core = torch.cat((state[..., :3] + state[..., 21:24] * .02 + dp,
                          next_q, state[..., 7:21] + state[..., 27:41] * .02 + dq,
                          state[..., 21:24] + dv, state[..., 24:27] + dw,
                          state[..., 27:41] + dqd, torque,
                          action, state[..., 55:69]), -1)
        if self.backlash:
            passive = torch.cat((state[..., 83:97] + state[..., 97:111] * .02 + delta[..., 54:68],
                                 state[..., 97:111] + delta[..., 68:82]), -1)
            return torch.cat((core, passive, state[..., 111:]), -1)
        return core


def world_error(prediction, target):
    # Quaternion sign invariant; group weighting prevents joint count domination.
    errors = [(prediction[..., :3] - target[..., :3]).square().mean(-1) / .03**2,
              (1 - (prediction[..., 3:7] * target[..., 3:7]).sum(-1).square().clamp(max=1)) / .2**2]
    for sl, scale in ((slice(7, 21), .3), (slice(21, 24), 1.), (slice(24, 27), 5.),
                      (slice(27, 41), 10.), (slice(41, 55), .3)):
        errors.append(((prediction[..., sl] - target[..., sl]) / scale).square().mean(-1))
    if prediction.shape[-1] > 83:
        errors.extend((((prediction[..., 83:97] - target[..., 83:97]) / .05).square().mean(-1),
                       ((prediction[..., 97:111] - target[..., 97:111]) / 5.).square().mean(-1)))
    return torch.stack(errors, -1).mean(-1)


class Critic(nn.Module):
    def __init__(self, state_dim=83, command_delay=False):
        super().__init__()
        self.command_delay = command_delay
        width = state_dim + (99 if command_delay else 0)
        self.norm = RunningNorm(width)
        self.net = mlp((width + 430, 256, 128, 1))

    def features(self, state, queue=None, length=None):
        if self.command_delay:
            return torch.cat((state, queue.flatten(-2), length[..., None].to(state.dtype)), -1)
        return state

    def forward(self, state, future, queue=None, length=None):
        return self.net(torch.cat((self.norm(self.features(state, queue, length)), goal_features(future)), -1)).squeeze(-1)


class ExportedActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, history, motion_latent):
        return self.actor.decode(history, motion_latent)
