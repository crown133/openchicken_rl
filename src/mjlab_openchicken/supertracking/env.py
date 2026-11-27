"""Pre-reset transitions and sensor-consistent references around the mjlab env."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab_openchicken.robot.openchicken_constants import OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG
from mjlab_openchicken.tasks import mdp
from .models import Kinematics
from .delay import BamCommandDelay


class Motion:
    def __init__(self, path, robot, device):
        self.path = Path(path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as a:
            self.joint_names, self.body_names = tuple(a['joint_names']), tuple(a['body_names'])
            servos = tuple(n for n in robot.joint_names if not n.startswith('passive_'))
            if self.joint_names != servos or any(n not in robot.body_names for n in self.body_names):
                raise ValueError('Motion joint/body ordering does not match training robot')
            if a['fps'] != 50 or a['quaternion_order'] != 'wxyz' or not np.allclose(a['qpos'][0], a['qpos'][-1]):
                raise ValueError('Expected closed, 50 Hz, wxyz motion')
            self.state = torch.tensor(np.c_[a['qpos'][:-1], a['body_lin_vel_w'][:-1, 0],
                                          a['body_ang_vel_w'][:-1, 0], a['joint_vel'][:-1]],
                                      dtype=torch.float32, device=device)
            self.pos = torch.tensor(a['body_pos_w'][:-1], dtype=torch.float32, device=device)
            self.quat = torch.tensor(a['body_quat_w'][:-1], dtype=torch.float32, device=device)
        if not torch.isfinite(self.state).all():
            raise ValueError('Nonfinite reference')

    def future(self, frame, length=10):
        indices = (frame[..., None] + 1 + torch.arange(length, device=frame.device)) % len(self.state)
        return self.state[indices]


class SupertrackingEnv:
    def __init__(self, env, motion):
        self.env, self.robot = env, env.scene['robot']
        self.device, self.n = env.device, env.num_envs
        self.motion = Motion(motion, self.robot, self.device)
        self.servo_ids = [self.robot.joint_names.index(n) for n in self.motion.joint_names]
        self.body_ids = [self.robot.body_names.index(n) for n in self.motion.body_names]
        backlash_names = {f'passive_{n}_backlash' for n in self.motion.joint_names}
        present = backlash_names.intersection(self.robot.joint_names)
        self.backlash = present == backlash_names
        expected = set(self.motion.joint_names) | {'passive_beak'}
        if self.backlash:
            expected |= backlash_names
        if len(self.servo_ids) != 14 or set(self.robot.joint_names) != expected:
            raise ValueError('Expected 14 servos, one beak, and optional 14 backlash hinges')
        self.passive_ids = ([self.robot.joint_names.index(f'passive_{n}_backlash')
                             for n in self.motion.joint_names] if self.backlash else None)
        if tuple(env.action_manager.get_term('joint_pos').target_names) != self.motion.joint_names:
            raise ValueError('Action order mismatch')
        # Coaxial backlash hinges have identical link FK at q_encoder=q_motor+q_play.
        # Use the original physical links, excluding inserted motor spacer bodies.
        nominal = Entity(deepcopy(OPENCHICKEN_ALLCOLLISIONS_ROBOT_CFG)).spec.compile()
        self.kinematics = Kinematics(nominal, [nominal.body(n).id for n in self.motion.body_names],
                                    [nominal.joint(n).id for n in self.motion.joint_names]).to(self.device)
        self.default = self.robot.data.default_joint_pos[0, self.servo_ids].clone()
        self.command_delay = BamCommandDelay(self.robot, env.cfg.decimation, env.physics_dt)
        self.limits = self.robot.data.joint_pos_limits[0, self.servo_ids].clone()
        self.frame = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode = torch.zeros_like(self.frame)
        self.age = torch.zeros_like(self.frame)
        self.commands = torch.zeros(self.n, 2, 14, device=self.device)
        self.history = torch.zeros(self.n, 10, 48, device=self.device)
        self.sensor_previous = torch.zeros(self.n, 48, device=self.device)
        self.noise = torch.tensor([.03]*3 + [.01]*3 + [.001]*14 + [.25]*14 + [0.]*14, device=self.device)
        if self.backlash:
            self.mount = mdp._imu_misalignment_quat(env, np.deg2rad(6.))
            self.lags = torch.randint(2, (self.n, 2), device=self.device).float()
            self.physics_context = self._physics_context()
        self.reset()
        self.state_dim = self.state().shape[-1]

    def _physics_context(self):
        model = self.env.sim.model
        body_ids = self.robot.indexing.body_ids[self.body_ids]
        joint_ids = self.robot.indexing.joint_ids[self.servo_ids].cpu().numpy()
        dofs = self.env.sim.mj_model.jnt_dofadr[joint_ids]
        feet, _ = self.robot.find_geoms(('left_foot_collision', 'right_foot_collision'))
        geom_ids = self.robot.indexing.geom_ids[feet]
        pieces = []
        for name, ids in (('body_mass', body_ids), ('body_inertia', body_ids), ('body_ipos', body_ids),
                          ('dof_armature', dofs), ('geom_friction', geom_ids)):
            value = getattr(model, name)[:, ids].flatten(1)
            pieces.append(value.expand(self.n, -1))
        actuator = self.robot.actuators[0]
        return torch.cat((*pieces, actuator.vin_tensor, actuator.vin_drop_gain, actuator.friction_scale), -1).clone()

    def state(self):
        state = mdp.supertracking_state(self.env, self.commands, self.passive_ids)
        if self.backlash:
            state = torch.cat((state, self.robot.data.encoder_bias[:, self.servo_ids],
                               self.mount, self.lags, self.physics_context), -1)
        return state.clone()

    def observation(self, state, noisy=True):
        obs, raw = mdp.supertracking_sensor_observation(state, self.default, self.sensor_previous,
                                                       self.noise if noisy else None)
        self.sensor_previous = raw
        return obs

    def set_iteration(self, iteration):
        if not self.backlash:
            return
        # Staged by this trainer's iterations, not the native PPO's 24-step epochs.
        trunk = .003 if iteration < 1500 else .005 if iteration < 3000 else .010 if iteration < 4500 else .015
        head = .003 if iteration < 1500 else .005 if iteration < 3000 else .010
        for name, magnitude in [('randomize_com', trunk), ('randomize_head_com', head)]:
            self.env.event_manager.get_term_cfg(name).params['ranges'] = (-magnitude, magnitude)

    @torch.no_grad()
    def reset(self, ids=None, frame=None, perturb=True):
        if ids is None:
            ids = torch.arange(self.n, device=self.device)
        with self.command_delay.reset_without_actuation():
            self.env.reset(env_ids=ids)
        if 'push_robot' in self.env.event_manager.active_terms.get('interval', []):
            # mjlab 1.3 resets class-event timers only; push_robot is a function.
            # Restart its timer explicitly so episodes/evaluations get 3–6 s.
            manager = self.env.event_manager
            index = manager.active_terms['interval'].index('push_robot')
            manager._interval_term_time_left[index][ids] = 3 + 3 * torch.rand(len(ids), device=self.device)
        if frame is None:
            frames = torch.randint(len(self.motion.state), (len(ids),), device=self.device)
            if self.backlash:
                frames[torch.rand(len(ids), device=self.device) < .3] = 0
            self.frame[ids] = frames
        else:
            self.frame[ids] = frame
        ref = self.motion.state[self.frame[ids]]
        root = torch.cat((ref[:, :7], ref[:, 21:27]), -1).clone()
        root[:, :3] += self.env.scene.env_origins[ids]
        joints = ref[:, 7:21].clone()
        if perturb:
            joints += torch.randn_like(joints) * .005
        joints = joints.clamp(self.limits[:, 0] + .005, self.limits[:, 1] - .005)
        q = torch.zeros(len(ids), len(self.robot.joint_names), device=self.device)
        qd = torch.zeros_like(q)
        q[:, self.servo_ids], qd[:, self.servo_ids] = joints, ref[:, 27:41]
        if self.backlash:
            play = (2 * torch.rand_like(joints) - 1) * np.deg2rad(1.) if perturb else torch.zeros_like(joints)
            q[:, self.passive_ids] = play
            q[:, self.servo_ids] -= play
            self.lags[ids] = torch.randint(2, (len(ids), 2), device=self.device).float()
            self.physics_context = self._physics_context()
        self.robot.write_joint_state_to_sim(q, qd, env_ids=ids)
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.robot.set_joint_position_target(joints, joint_ids=self.servo_ids, env_ids=ids[:, None])
        self.command_delay.seed(ids, joints)
        self.commands[ids] = 0
        self.env.sim.forward()
        self.env.sim.sense()
        self.episode[ids] += 1
        self.age[ids] = 0
        # Only replace reset environments; ongoing environments keep sensor history.
        obs, raw = mdp.supertracking_sensor_observation(self.state(), self.default,
                                                       noise=self.noise if perturb else None)
        self.sensor_previous[ids] = raw[ids]
        self.history[ids] = obs[ids, None]

    @torch.no_grad()
    def step(self, action, reset=True, noisy=True):
        if not torch.isfinite(action).all():
            raise FloatingPointError('Nonfinite action')
        action = action.clamp(self.limits[:, 0] + .01 - self.default,
                              self.limits[:, 1] - .01 - self.default)
        transition = {'state': self.state(), 'history': self.history.clone(),
                      'sensor_previous': self.sensor_previous.clone(),
                      'frame': self.frame.clone(), 'episode': self.episode.clone(), 'action': action.clone()}
        transition['command_queue'], transition['command_length'] = self.command_delay.snapshot()
        with self.command_delay.record():
            self.env.step(action)
        transition['command_lags'], transition['applied_targets'] = self.command_delay.transition()
        if 'push_robot' in self.env.event_manager.active_terms.get('interval', []):
            # Interval events modify qvel after mjlab's forward(). Align derived
            # velocities with the post-push state used by independent tests.
            self.env.sim.forward()
            self.env.sim.sense()
        self.commands = torch.stack((action, self.commands[:, 0]), 1)
        self.frame = (self.frame + 1) % len(self.motion.state)
        self.age += 1
        if self.backlash:
            update = self.age % 64 == 0
            self.lags[update] = torch.randint(2, (int(update.sum()), 2), device=self.device).float()
        state = self.state()
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite physics state; transition was not stored')
        storage, data = self.env.sim.wp_data, self.env.sim.data
        if bool(((data.ncollision > storage.naconmax) | (data.nacon > storage.naconmax)).any()) or bool((data.nefc > storage.njmax).any()):
            raise RuntimeError('MuJoCo Warp contact/constraint capacity exceeded')
        pos = self.robot.data.body_link_pos_w[:, self.body_ids] - self.env.scene.env_origins[:, None]
        quat = self.robot.data.body_link_quat_w[:, self.body_ids]
        ref = self.motion.state[self.frame]
        reward, errors = mdp.supertracking_reward(state, ref, pos, quat, self.motion.pos[self.frame],
                                                 self.motion.quat[self.frame], action, transition['state'][:, 55:69])
        failed = mdp.supertracking_failure(state, ref)
        timeout = self.age >= len(self.motion.state)
        done = failed | timeout
        transition.update(next_state=state, reward=reward, done=done, failed=failed)
        self.history = torch.cat((self.history[:, 1:], self.observation(state, noisy)[:, None]), 1)
        lengths = self.age[done].clone()
        if reset and done.any():
            self.reset(done.nonzero().flatten())
        return transition, errors, lengths
