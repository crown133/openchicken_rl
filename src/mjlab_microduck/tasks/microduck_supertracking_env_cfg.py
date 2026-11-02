"""V12 model-based tracking foundation, nominal-physics first training stage."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from mjlab.rl.config import RslRlBaseRunnerCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from .backlash import make_backlash_variant

_MOTIONS = Path(__file__).resolve().parents[3] / "motions"
DEFAULT_MOTION = str(_MOTIONS / "v12/motion.npz")
ROBUST_MOTION = str(_MOTIONS / "v12_stand_transitions/motion.npz")


def make_microduck_supertracking_env_cfg(play=False, rough=False, robust=False, push_test=False):
    if rough:
        raise ValueError("V12 uses a fixed stance on flat ground")
    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)}
    # Keep the inherited 61D observations and their noise/delays intact. The
    # model-based runner constructs its explicit 48D history separately.
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    events = {"expand_bam_friction_fields", "reset_action_history"}
    if robust:
        events.update(("foot_friction", "encoder_bias", "randomize_com", "randomize_head_com",
                       "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"))
        if push_test:
            events.add('push_robot')
    cfg.events = {k: v for k, v in cfg.events.items() if k in events}
    if robust:
        cfg = make_backlash_variant(cfg)
        cfg.scene.entities['robot'] = deepcopy(cfg.scene.entities['robot'])
        # bearing_roll is the right hip, not a head component.
        head = cfg.events['randomize_head_com'].params['asset_cfg']
        head.body_names = tuple(n for n in head.body_names if n != 'bearing_roll')
        cfg.events['randomize_armature'].params['asset_cfg'].joint_names = (r'^(?!passive_).*',)
        if push_test:
            cfg.events['push_robot'].interval_range_s = (3., 6.)
            cfg.events['push_robot'].params['velocity_range'] = {'x': (-.30, .30), 'y': (-.30, .30)}
    cfg.auto_reset = False
    cfg.episode_length_s = 12. if robust else 10.
    cfg.scene.num_envs = 256
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 1024
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.vin_range = (6.5, 8.2) if robust else (7.4, 7.4)
    actuator.vin_drop_gain_range = (0., .2) if robust else (.1, .1)
    # BAM's stochastic 3–6 physics-step delay and all collision masks remain.
    return cfg


@dataclass
class MicroduckSupertrackingRlCfg(RslRlBaseRunnerCfg):
    experiment_name: str = "supertracking_v12"
    motion: str = DEFAULT_MOTION
    robust: bool = False
    max_iterations: int = 3000
    num_steps_per_env: int = 8
    save_interval: int = 100
    capacity: int = 256
    batch_size: int = 128
    wm_batches: int = 4
    wm_window: int = 8
    horizon: int = 32
    warmup_steps: int = 40
    wm_only_iters: int = 30
    learning_rate: float = 2e-4
    eval_interval: int = 100
    upload_model: bool = False
