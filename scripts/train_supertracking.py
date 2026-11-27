"""Train OpenChicken V12 with a learned world model and differentiable imagination."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_openchicken.tasks.openchicken_supertracking_env_cfg import (
    OpenChickenSupertrackingRlCfg, make_openchicken_supertracking_env_cfg, ROBUST_MOTION,
)
from mjlab_openchicken.supertracking.runner import SupertrackingRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--motion')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--wm-batches', type=int)
    parser.add_argument('--replay-device', choices=('cpu', 'cuda:0'), help='Replay storage only; training stays on --device')
    parser.add_argument('--constraint-capacity', type=int, help='MuJoCo per-environment constraint allocation')
    parser.add_argument('--contact-capacity', type=int, help='MuJoCo contact allocation per environment in the shared pool')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--robust', action='store_true', help='Backlash, domain randomization and delayed sensors; 12s reference')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = asdict(OpenChickenSupertrackingRlCfg())
    num_envs = 256
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        cfg = checkpoint['cfg']
        cfg.setdefault('robust', False)
        num_envs = checkpoint['provenance']['num_envs']
        del checkpoint
    elif args.output.exists():
        raise FileExistsError('Use a new output directory, or --resume an existing checkpoint')
    if args.smoke:
        cfg.update(max_iterations=5, wm_only_iters=1, wm_batches=2, batch_size=32,
                   eval_interval=5, save_interval=5, smoke=True)
        num_envs = 64
    if args.robust:
        cfg['robust'] = True
        if not args.resume:
            cfg['motion'] = ROBUST_MOTION
    for argument, key in [('iterations', 'max_iterations'), ('batch_size', 'batch_size'),
                          ('wm_batches', 'wm_batches'), ('motion', 'motion'),
                          ('replay_device', 'replay_device'), ('constraint_capacity', 'constraint_capacity'),
                          ('contact_capacity', 'contact_capacity')]:
        value = getattr(args, argument)
        if value is not None:
            cfg[key] = value
    num_envs = args.num_envs or num_envs
    if num_envs < 8:
        raise ValueError('At least eight environments are needed for held-out validation')
    torch.manual_seed(cfg['seed'])
    env_cfg = make_openchicken_supertracking_env_cfg(robust=cfg['robust'])
    if cfg.get('constraint_capacity') is not None:
        env_cfg.sim.njmax = cfg['constraint_capacity']
    if cfg.get('contact_capacity') is not None:
        env_cfg.sim.nconmax = cfg['contact_capacity']
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    try:
        runner = SupertrackingRunner(env, cfg, args.output, args.device)
        if args.resume:
            runner.load(args.resume)
        (args.output / 'process.json').write_text(json.dumps({'pid': os.getpid(), 'argv': sys.argv, 'status': 'running'}, indent=2))
        runner.learn(cfg['max_iterations'])
        (args.output / 'process.json').write_text(json.dumps({'pid': os.getpid(), 'iteration': runner.iteration, 'status': 'completed'}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
