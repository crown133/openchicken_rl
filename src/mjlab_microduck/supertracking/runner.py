"""Online world-model fitting and BPTT actor/critic training, without PPO."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_supertracking_env_cfg import MicroduckSupertrackingRlCfg
from .env import SupertrackingEnv
from .delay import command_target, delayed_targets
from .models import Actor, Critic, WorldModel, ExportedActor, goal_features, world_error, normalize_quat, rotation_quat


class Replay:
    def __init__(self, capacity, num_envs, device, storage_device=None):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        self.storage_device = storage_device or device
        self.pos = self.size = 0
        self.data = {}

    def add(self, transition):
        for key, value in transition.items():
            if key not in self.data:
                self.data[key] = torch.zeros((self.capacity, *value.shape), dtype=value.dtype, device=self.storage_device)
            self.data[key][self.pos].copy_(value)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, length, validation=False):
        if self.size < length:
            raise ValueError('Insufficient consecutive transitions')
        heldout = max(1, self.num_envs // 8)
        low, high = (self.num_envs - heldout, self.num_envs) if validation else (0, self.num_envs - heldout)
        env = torch.randint(low, high, (batch,), device=self.device)
        start = torch.randint(self.size - length + 1, (batch,), device=self.device)
        times = (self.pos - self.size + start[:, None] + torch.arange(length, device=self.device)) % self.capacity
        storage_times, storage_env = times.to(self.storage_device), env.to(self.storage_device)
        result = {k: v[storage_times, storage_env[:, None]].to(self.device) for k, v in self.data.items()}
        result['mask'] = result['episode'] == result['episode'][:, :1]
        return result


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def optimize(loss, optimizer, module):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


class SupertrackingRunner:
    def __init__(self, env, train_cfg=None, log_dir=None, device='cuda:0', **kwargs):
        base = env.unwrapped if hasattr(env, 'unwrapped') else env
        defaults = asdict(MicroduckSupertrackingRlCfg())
        defaults.update(train_cfg or {})
        self.cfg, self.device = defaults, device
        self.output = Path(log_dir or 'logs/supertracking_v12')
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = SupertrackingEnv(base, defaults['motion'])
        self.actor = Actor(self.env.default, self.env.limits).to(device)
        self.world = WorldModel(self.env.state_dim, self.env.backlash, command_delay=True).to(device)
        self.critic = Critic(self.env.state_dim, command_delay=True).to(device)
        lr = defaults['learning_rate']
        self.optimizers = {k: torch.optim.Adam(m.parameters(), lr=lr) for k, m in self.modules.items()}
        self.replay = Replay(defaults['capacity'], self.env.n, device, defaults.get('replay_device'))
        self.iteration = 0
        self.best = (-1., -1.)
        self.smoke = defaults.get('smoke', False)
        self.started = time.monotonic()
        with torch.no_grad():
            frames = torch.arange(len(self.env.motion.state), device=device)
            self.actor.goal_norm.update(goal_features(self.env.motion.future(frames)))
        self.provenance = {'motion_sha256': self.env.motion.sha256,
                           'reference_implementation': 'https://github.com/Breezewrf/supermimic_mjlab/tree/a7cee74636a4c596cd8710f8c2e51f82d78bb150',
                           'algorithm': '8-step autoregressive world model; 32-step BPTT actor and lambda-return critic',
                           'policy_schema': 'microduck_supertracking_v1: history[10,48] + latent[32] -> HOME-relative action[14]',
                           'physics_stage': 'nominal 7.4V, sag .1, original stochastic BAM 3-6 substep delay; sensor/reset noise',
                           'num_envs': self.env.n, 'device': device,
                           'checkpoint_resume': 'optimizer/replay/RNG restored; simulation starts new episodes (not bitwise continuation)'}
        self.provenance.update(state_dim=self.env.state_dim, backlash=self.env.backlash,
                               motion_frames=len(self.env.motion.state), evaluation_seed=20261101,
                               command_delay_schema='bam_substeps_v1',
                               command_delay='7x14 absolute-target queue; 4 independent shared-joint lags in [3,6]; reset posture prefill')
        if self.env.backlash:
            self.provenance['physics_stage'] = (
                'BAM M6 6.5-8.2V, sag 0-.2; +/-1deg mechanical backlash; '
                'friction/mass/inertia/CoM/armature DR; encoder bias +/-0.015rad; '
                'IMU mounting 0-6deg; IMU 0-20ms, joint velocity 20ms delay; '
                'no training velocity kicks; independent push-test configuration available')
        package = Path(__file__).resolve().parents[1]
        sources = [*Path(__file__).parent.glob('*.py'), package / 'tasks/mdp.py',
                   package / 'tasks/microduck_supertracking_env_cfg.py', package / 'tasks/__init__.py']
        self.provenance['source_sha256'] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (self.output / 'config.json').write_text(json.dumps(defaults, indent=2))
        (self.output / 'provenance.json').write_text(json.dumps(self.provenance, indent=2))

    @property
    def modules(self):
        return {'actor': self.actor, 'world': self.world, 'critic': self.critic}

    def add_git_repo_to_log(self, path):
        # mjlab's generic train entry invokes this hook. Record the launcher
        # identity without changing or initializing any Git metadata.
        p = Path(path).resolve()
        (self.output / 'launcher.json').write_text(json.dumps(
            {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}, indent=2))

    @torch.no_grad()
    def collect(self, steps):
        rewards, failures, lengths = [], [], []
        for _ in range(steps):
            future = self.env.motion.future(self.env.frame)
            action = self.actor(self.env.history, future)
            std = max(.02, .07 * (1 - self.iteration / max(self.cfg['max_iterations'], 1)))
            action += torch.randn_like(action) * std
            transition, errors, completed = self.env.step(action)
            self.replay.add(transition)
            # Update each normalization once per real transition, never from dreams.
            self.world.norm.update(self.world.features(transition['state'], transition['action'], transition['applied_targets']))
            self.actor.obs_norm.update(transition['history'][:, -1])
            self.critic.norm.update(self.critic.features(transition['state'], transition['command_queue'], transition['command_length']))
            rewards.append(transition['reward'].mean())
            failures.append(transition['failed'].float().mean())
            if len(completed):
                lengths.append(completed.float().mean())
        return {'real_reward': float(torch.stack(rewards).mean()),
                'failure_fraction': float(torch.stack(failures).mean()),
                'finished_episode_seconds': float(torch.stack(lengths).mean() / 50) if lengths else None,
                **{k: float(v.mean()) for k, v in errors.items()}}

    def world_loss(self, batch, baseline=False):
        prediction = batch['state'][:, 0]
        queue, length = batch['command_queue'][:, 0], batch['command_length'][:, 0]
        losses = []
        for step in range(batch['action'].shape[1]):
            prediction = torch.where(batch['mask'][:, step, None], prediction, batch['state'][:, step])
            if baseline:
                # Measured constant-velocity baseline, not a zero-valued target.
                prediction = prediction.clone()
                prediction[:, :3] = prediction[:, :3] + prediction[:, 21:24] * .02
                prediction[:, 7:21] = prediction[:, 7:21] + prediction[:, 27:41] * .02
                if self.env.backlash:
                    prediction[:, 83:97] = prediction[:, 83:97] + prediction[:, 97:111] * .02
                q = prediction[:, 3:7].clone()
                prediction[:, 3:7] = normalize_quat(quat_mul(q, rotation_quat(
                    quat_apply_inverse(q, prediction[:, 24:27]) * .02)))
            else:
                queue = torch.where(batch['mask'][:, step, None, None], queue, batch['command_queue'][:, step])
                length = torch.where(batch['mask'][:, step], length, batch['command_length'][:, step])
                applied, queue, length = delayed_targets(
                    queue, length, command_target(prediction, batch['action'][:, step], self.env.default), batch['command_lags'][:, step])
                prediction = self.world(prediction, batch['action'][:, step], applied)
            losses.append(world_error(prediction, batch['next_state'][:, step]))
        errors = torch.stack(losses, 1)
        return errors[batch['mask']].mean()

    def update_world(self):
        loss = None
        for _ in range(self.cfg['wm_batches']):
            batch = self.replay.sample(self.cfg['batch_size'], self.cfg['wm_window'])
            loss = self.world_loss(batch)
            optimize(loss, self.optimizers['world'], self.world)
        with torch.no_grad():
            batch = self.replay.sample(self.cfg['batch_size'], self.cfg['wm_window'], validation=True)
            validation = float(self.world_loss(batch))
            baseline = float(self.world_loss(batch, baseline=True))
        metrics = {'world_loss': float(loss.detach()), 'world_validation_8step': validation,
                   'world_baseline_8step': baseline}
        if self.smoke or self.iteration % self.cfg['eval_interval'] == 0:
            with torch.no_grad():
                for length in (1, 32):
                    batch = self.replay.sample(self.cfg['batch_size'], length, validation=True)
                    metrics[f'world_validation_{length}step'] = float(self.world_loss(batch))
                    metrics[f'world_baseline_{length}step'] = float(self.world_loss(batch, baseline=True))
        return metrics

    def imagine(self, starts):
        state = starts['state'][:, 0].detach()
        queue, length = starts['command_queue'][:, 0].detach(), starts['command_length'][:, 0]
        history = starts['history'][:, 0].detach()
        sensor_previous = (starts['sensor_previous'][:, 0].detach() if 'sensor_previous' in starts
                           else history[:, -1])
        frame = starts['frame'][:, 0]
        horizon = self.cfg['horizon']
        frames = (frame[:, None] + torch.arange(horizon + 1, device=self.device)) % len(self.env.motion.state)
        future = self.env.motion.future(frames)
        latent = self.actor.encode(future)
        states, actions, queues, lengths = [state], [], [queue], [length]
        for t in range(horizon):
            action = self.actor.decode(history, latent[:, t])
            applied, queue, length = delayed_targets(
                queue, length, command_target(state, action, self.env.default), self.env.command_delay.sample(len(state)))
            state = self.world(state, action, applied)
            obs, sensor_previous = mdp.supertracking_sensor_observation(
                state, self.env.default, sensor_previous, self.env.noise if self.env.backlash else None)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            actions.append(action)
            states.append(state)
            queues.append(queue)
            lengths.append(length)
        states, actions = torch.stack(states, 1), torch.stack(actions, 1)
        queues, lengths = torch.stack(queues, 1), torch.stack(lengths, 1)
        next_frames = frames[:, 1:]
        next_states = states[:, 1:]
        body_pos, body_quat = self.env.kinematics(next_states)
        reward, _ = mdp.supertracking_reward(next_states, self.env.motion.state[next_frames], body_pos, body_quat,
                                             self.env.motion.pos[next_frames], self.env.motion.quat[next_frames],
                                             actions, states[:, :-1, 55:69])
        continuation = 1 - mdp.supertracking_failure(next_states, self.env.motion.state[next_frames], soft=True)
        survival = continuation.cumprod(1)
        before = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), 1)
        discount = .99 ** torch.arange(horizon, device=self.device)
        terminal_value = self.critic(states[:, -1], future[:, -1], queues[:, -1], lengths[:, -1])
        objective = ((discount * before * reward).sum(1) + .99**horizon * survival[:, -1] * terminal_value).mean()
        with torch.no_grad():
            values = self.critic(states.detach(), future, queues.detach(), lengths)
            target = values[:, -1]
            targets = []
            for t in reversed(range(horizon)):
                target = reward[:, t].detach() + .99 * continuation[:, t].detach() * (.05 * values[:, t + 1] + .95 * target)
                targets.append(target)
            targets = torch.stack(targets[::-1], 1)
        return (objective, states[:, :-1].detach(), future[:, :-1], targets, before.detach(),
                queues[:, :-1].detach(), lengths[:, :-1])

    def update_policy(self):
        starts = self.replay.sample(self.cfg['batch_size'], 1)
        with frozen(self.world, self.critic):
            objective, states, future, target, weight, queue, length = self.imagine(starts)
            norm = optimize(-objective / self.cfg['horizon'], self.optimizers['actor'], self.actor)
        critic_loss = None
        for _ in range(2):
            error = (self.critic(states, future, queue, length) - target).square()
            critic_loss = (error * weight).sum() / weight.sum().clamp_min(1)
            optimize(critic_loss, self.optimizers['critic'], self.critic)
        return {'imagined_objective': float(objective.detach()), 'actor_grad_norm': norm,
                'critic_loss': float(critic_loss.detach())}

    @torch.no_grad()
    def evaluate(self):
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261101)
            self.env.set_iteration(4500)  # Compare every checkpoint at the full DR envelope.
            result = self._evaluate()
        self.env.set_iteration(self.iteration)
        self.env.reset()
        return result

    @torch.no_grad()
    def _evaluate(self):
        self.env.reset(frame=0, perturb=False)
        alive = torch.ones(self.env.n, dtype=torch.bool, device=self.device)
        steps = torch.zeros(self.env.n, device=self.device)
        rewards = torch.zeros_like(steps)
        joint_error = torch.zeros_like(steps)
        root_error = torch.zeros_like(steps)
        duration = len(self.env.motion.state)
        trace = [self.env.state()[0].cpu().numpy()]
        for _ in range(duration):
            action = self.actor(self.env.history, self.env.motion.future(self.env.frame))
            tr, errors, _ = self.env.step(action, reset=False, noisy=self.env.backlash)
            rewards += tr['reward'] * alive
            joint_error += errors['joint_rmse_rad'] * alive
            root_error += errors['root_error_m'] * alive
            steps += alive
            alive &= ~tr['failed']
            trace.append(tr['next_state'][0].cpu().numpy())
            if not alive.any():
                break
        result = {'success_fraction': float(alive.float().mean()),
                  'completion_fraction': float(steps.mean() / duration),
                  'normalized_return': float(rewards.mean() / duration),
                  'duration_s': duration / 50,
                  'surviving_steps_joint_rmse_deg': float(joint_error.sum() / steps.sum().clamp_min(1) * 180 / np.pi),
                  'surviving_steps_root_error_mm': float(root_error.sum() / steps.sum().clamp_min(1) * 1000)}
        np.savez_compressed(self.output / 'evaluation_trace.npz', state=np.stack(trace), dt=.02)
        (self.output / 'evaluation.json').write_text(json.dumps({'iteration': self.iteration, **result}, indent=2))
        return result

    def save(self, path):
        payload = {'iteration': self.iteration, 'cfg': self.cfg, 'provenance': self.provenance,
                   'modules': {k: v.state_dict() for k, v in self.modules.items()},
                   'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
                   'replay': {'data': self.replay.data, 'pos': self.replay.pos, 'size': self.replay.size},
                   'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   'best': self.best}
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path, **kwargs):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['provenance'].get('command_delay_schema') != self.provenance['command_delay_schema']:
            raise ValueError('Checkpoint predates explicit BAM command transport; retrain the world model with a fresh replay')
        if checkpoint['provenance']['motion_sha256'] != self.env.motion.sha256:
            raise ValueError('Checkpoint/reference mismatch')
        if checkpoint['provenance']['num_envs'] != self.env.n:
            raise ValueError('Resume must keep the replay environment count')
        if checkpoint['cfg'].get('robust', False) != self.cfg['robust']:
            raise ValueError('Resume must keep the physics/sensor contract')
        for key in self.modules:
            self.modules[key].load_state_dict(checkpoint['modules'][key])
            self.optimizers[key].load_state_dict(checkpoint['optimizers'][key])
        self.replay.data = {k: v.to(self.replay.storage_device) for k, v in checkpoint['replay']['data'].items()}
        self.replay.pos, self.replay.size = checkpoint['replay']['pos'], checkpoint['replay']['size']
        self.iteration, self.best = checkpoint['iteration'], checkpoint['best']
        self.env.set_iteration(self.iteration)
        self.env.episode[:] = self.replay.data['episode'].max() + 1
        torch.set_rng_state(checkpoint['rng'].cpu())
        if checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])

    @torch.no_grad()
    def export(self):
        actor = self.actor.cpu().eval()
        module = ExportedActor(actor)
        history = torch.zeros(2, 10, 48)
        latent = torch.zeros(2, 32)
        torch.onnx.export(module, (history, latent), str(self.output / 'policy.onnx'),
                          input_names=['history', 'motion_latent'], output_names=['action'],
                          dynamic_axes={'history': {0: 'batch'}, 'motion_latent': {0: 'batch'}, 'action': {0: 'batch'}},
                          opset_version=17, dynamo=False)
        import onnxruntime as ort
        session = ort.InferenceSession(str(self.output / 'policy.onnx'), providers=['CPUExecutionProvider'])
        history = torch.randn(2, 10, 48)
        future = self.env.motion.future(torch.arange(len(self.env.motion.state), device=self.device)).cpu()
        latents = actor.encode(future)
        latent = latents[:2]
        actual = session.run(None, {'history': history.numpy(), 'motion_latent': latent.numpy()})[0]
        np.testing.assert_allclose(actual, module(history, latent).numpy(), atol=2e-5, rtol=2e-5)
        np.savez_compressed(self.output / 'motion_latents_50hz.npz', latent=latents.numpy(), fps=50)
        self.actor.to(self.device).train()
        (self.output / 'policy_interface.json').write_text(json.dumps({
            'schema': 'microduck_supertracking_v1', 'history': [10, 48], 'motion_latent': [32], 'action': [14],
            'proprioception_order': ['body_angular_velocity(3)', 'projected_gravity(3)', 'joint_pos_minus_HOME(14)', 'joint_vel(14)', 'previous_HOME_relative_command(14)'],
            'normalization': 'RunningNorm included inside ONNX', 'action_units': 'radians relative to HOME',
            'joint_names': self.env.motion.joint_names, 'HOME': self.env.default.cpu().tolist(),
            'backlash_encoder_view': self.env.backlash,
            'duration_s': len(self.env.motion.state) / 50,
            'motion_sha256': self.env.motion.sha256, 'runtime_61D_compatible': False,
            'hardware_deployment_ready': False}, indent=2))

    def learn(self, num_learning_iterations=None, init_at_random_ep_len=False):
        total = num_learning_iterations or self.cfg['max_iterations']
        if self.replay.size < self.cfg['warmup_steps']:
            self.collect(self.cfg['warmup_steps'] - self.replay.size)
        with (self.output / 'metrics.jsonl').open('a', buffering=1) as log:
            while self.iteration < total:
                self.iteration += 1
                self.env.set_iteration(self.iteration)
                metrics = self.collect(self.cfg['num_steps_per_env'])
                metrics.update(self.update_world())
                ready = metrics['world_validation_8step'] < metrics['world_baseline_8step']
                if self.iteration > self.cfg['wm_only_iters'] and (ready or self.smoke):
                    metrics.update(self.update_policy())
                    metrics['stage'] = 'world_and_policy'
                else:
                    metrics['stage'] = 'world_model_warmup'
                if self.iteration % self.cfg['eval_interval'] == 0 or self.iteration == total:
                    metrics['evaluation'] = self.evaluate()
                    score = (metrics['evaluation']['success_fraction'], metrics['evaluation']['normalized_return'])
                    if score > self.best:
                        self.best = score
                        self.save(self.output / 'best.pt')
                metrics.update(iteration=self.iteration, elapsed_s=time.monotonic() - self.started)
                log.write(json.dumps(metrics) + '\n')
                print(json.dumps(metrics), flush=True)
                if self.iteration % self.cfg['save_interval'] == 0 or self.iteration == total:
                    self.save(self.output / 'last.pt')
            self.export()
