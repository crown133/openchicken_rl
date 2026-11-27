from mjlab_openchicken.train_hook import maybe_init_checkpoint, maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()
# `--init-checkpoint <path | hf://bundle>`: resume/warm-start from a shared training bundle
# (checkpoints.py, docs/training_checkpoints.md). A no-op without the flag.
maybe_init_checkpoint()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class OpenChickenOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .openchicken_velocity_env_cfg import (
    make_openchicken_velocity_env_cfg,
    OpenChickenRlCfg,
)
from .openchicken_standup_env_cfg import (
    make_openchicken_standup_env_cfg,
    OpenChickenStandUpRlCfg,
)
from .openchicken_velstand_env_cfg import (
    make_openchicken_velstand_env_cfg,
    OpenChickenVelStandRlCfg,
)
from .openchicken_ground_pick_env_cfg import (
    make_openchicken_ground_pick_env_cfg,
    OpenChickenGroundPickRlCfg,
)
from .openchicken_ball_kick_env_cfg import (
    make_openchicken_ball_kick_env_cfg,
    OpenChickenBallKickRlCfg,
)
from .openchicken_sitstand_env_cfg import (
    make_openchicken_sitstand_env_cfg,
    OpenChickenSitStandRlCfg,
)
from .openchicken_velocity_rollers_env_cfg import (
    make_openchicken_velocity_rollers_env_cfg,
    OpenChickenRollersRlCfg,
)
from .openchicken_velocity_swizzle_env_cfg import (
    make_openchicken_velocity_swizzle_env_cfg,
    OpenChickenSwizzleRlCfg,
)
from .openchicken_roller_crouch_env_cfg import (
    make_openchicken_roller_crouch_env_cfg,
    OpenChickenRollerCrouchRlCfg,
)
from .openchicken_roller_slope_env_cfg import (
    make_openchicken_roller_slope_env_cfg,
    OpenChickenRollerSlopeRlCfg,
)
from .openchicken_roller_standup_env_cfg import (
    make_openchicken_roller_standup_env_cfg,
    OpenChickenRollerStandUpRlCfg,
)
from .openchicken_spin_env_cfg import (
    make_openchicken_spin_env_cfg,
    OpenChickenSpinRlCfg,
)
from .openchicken_roulade_env_cfg import (
    make_openchicken_roulade_env_cfg,
    OpenChickenRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-OpenChicken",
    env_cfg=make_openchicken_velocity_env_cfg(),
    play_env_cfg=make_openchicken_velocity_env_cfg(play=True),
    rl_cfg=OpenChickenRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-OpenChicken",
    env_cfg=make_openchicken_velocity_env_cfg(rough=True),
    play_env_cfg=make_openchicken_velocity_env_cfg(play=True, rough=True),
    rl_cfg=OpenChickenRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-OpenChicken",
    env_cfg=make_openchicken_velstand_env_cfg(),
    play_env_cfg=make_openchicken_velstand_env_cfg(play=True),
    rl_cfg=OpenChickenVelStandRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-OpenChicken",
    env_cfg=make_openchicken_velstand_env_cfg(rough=True),
    play_env_cfg=make_openchicken_velstand_env_cfg(play=True, rough=True),
    rl_cfg=OpenChickenVelStandRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-OpenChicken",
    env_cfg=make_openchicken_standup_env_cfg(),
    play_env_cfg=make_openchicken_standup_env_cfg(play=True),
    rl_cfg=OpenChickenStandUpRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-OpenChicken",
    env_cfg=make_openchicken_standup_env_cfg(rough=True),
    play_env_cfg=make_openchicken_standup_env_cfg(play=True, rough=True),
    rl_cfg=OpenChickenStandUpRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-OpenChicken",
    env_cfg=make_openchicken_sitstand_env_cfg(),
    play_env_cfg=make_openchicken_sitstand_env_cfg(play=True),
    rl_cfg=OpenChickenSitStandRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-OpenChicken",
    env_cfg=make_openchicken_sitstand_env_cfg(rough=True),
    play_env_cfg=make_openchicken_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=OpenChickenSitStandRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-OpenChicken",
    env_cfg=make_openchicken_ground_pick_env_cfg(),
    play_env_cfg=make_openchicken_ground_pick_env_cfg(play=True),
    rl_cfg=OpenChickenGroundPickRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-OpenChicken",
    env_cfg=make_openchicken_ball_kick_env_cfg(),
    play_env_cfg=make_openchicken_ball_kick_env_cfg(play=True),
    rl_cfg=OpenChickenBallKickRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-OpenChicken",
    env_cfg=make_openchicken_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_openchicken_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=OpenChickenGroundPickRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-OpenChicken-Rollers",
    env_cfg=make_openchicken_velocity_rollers_env_cfg(),
    play_env_cfg=make_openchicken_velocity_rollers_env_cfg(play=True),
    rl_cfg=OpenChickenRollersRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-OpenChicken",
    env_cfg=make_openchicken_velocity_swizzle_env_cfg(),
    play_env_cfg=make_openchicken_velocity_swizzle_env_cfg(play=True),
    rl_cfg=OpenChickenSwizzleRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-OpenChicken",
    env_cfg=make_openchicken_roller_crouch_env_cfg(),
    play_env_cfg=make_openchicken_roller_crouch_env_cfg(play=True),
    rl_cfg=OpenChickenRollerCrouchRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-OpenChicken",
    env_cfg=make_openchicken_roller_slope_env_cfg(),
    play_env_cfg=make_openchicken_roller_slope_env_cfg(play=True),
    rl_cfg=OpenChickenRollerSlopeRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-OpenChicken",
    env_cfg=make_openchicken_roller_standup_env_cfg(),
    play_env_cfg=make_openchicken_roller_standup_env_cfg(play=True),
    rl_cfg=OpenChickenRollerStandUpRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-OpenChicken",
    env_cfg=make_openchicken_spin_env_cfg(),
    play_env_cfg=make_openchicken_spin_env_cfg(play=True),
    rl_cfg=OpenChickenSpinRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-OpenChicken",
    env_cfg=make_openchicken_roulade_env_cfg(),
    play_env_cfg=make_openchicken_roulade_env_cfg(play=True),
    rl_cfg=OpenChickenRouladeRlCfg,
    runner_cls=OpenChickenOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_openchicken.robot.openchicken_constants import (
    OPENCHICKEN_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    OPENCHICKEN_BACKLASH_ROBOT_CFG,
    OPENCHICKEN_ROLLERS_BACKLASH_ROBOT_CFG,
    OPENCHICKEN_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = OPENCHICKEN_BACKLASH_ROBOT_CFG
_BL_WALK = OPENCHICKEN_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = OPENCHICKEN_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = OPENCHICKEN_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-OpenChicken", make_openchicken_velocity_env_cfg, {}, OpenChickenRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-OpenChicken", make_openchicken_velocity_env_cfg, {"rough": True}, OpenChickenRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-OpenChicken", make_openchicken_velstand_env_cfg, {}, OpenChickenVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-OpenChicken", make_openchicken_velstand_env_cfg, {"rough": True}, OpenChickenVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-OpenChicken", make_openchicken_standup_env_cfg, {}, OpenChickenStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-OpenChicken", make_openchicken_standup_env_cfg, {"rough": True}, OpenChickenStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-OpenChicken", make_openchicken_sitstand_env_cfg, {}, OpenChickenSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-OpenChicken", make_openchicken_sitstand_env_cfg, {"rough": True}, OpenChickenSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-OpenChicken", make_openchicken_ground_pick_env_cfg, {}, OpenChickenGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-OpenChicken", make_openchicken_ground_pick_env_cfg, {"rough": True}, OpenChickenGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-OpenChicken", make_openchicken_ball_kick_env_cfg, {}, OpenChickenBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-OpenChicken-Rollers", make_openchicken_velocity_rollers_env_cfg, {}, OpenChickenRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-OpenChicken", make_openchicken_velocity_swizzle_env_cfg, {}, OpenChickenSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-OpenChicken", make_openchicken_roller_crouch_env_cfg, {}, OpenChickenRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-OpenChicken", make_openchicken_roller_slope_env_cfg, {}, OpenChickenRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=OpenChickenOnPolicyRunner,
    )

# Independent model-based runner; original 61D PPO tasks are unchanged.
# Backlash is deliberately unsupported until its passive joint state is modeled.
from .openchicken_supertracking_env_cfg import (
    make_openchicken_supertracking_env_cfg, OpenChickenSupertrackingRlCfg,
)
from mjlab_openchicken.supertracking.runner import SupertrackingRunner

register_mjlab_task(
    task_id="Mjlab-Supertracking-Flat-OpenChicken",
    env_cfg=make_openchicken_supertracking_env_cfg(),
    play_env_cfg=make_openchicken_supertracking_env_cfg(play=True),
    rl_cfg=OpenChickenSupertrackingRlCfg(),
    runner_cls=SupertrackingRunner,
)
