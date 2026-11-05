"""Export a local Supertracking checkpoint on CPU through the training exporter."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Initialize task registration before importing the task-dependent helpers.
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.supertracking.env import Motion
from mjlab_microduck.supertracking.export import sha256
from mjlab_microduck.supertracking.models import Actor
from mjlab_microduck.supertracking.runner import SupertrackingRunner


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--motion', type=Path, default=Path(
        'motion/assets/v12_stand_transitions/motion.npz'))
    parser.add_argument('--interface', type=Path,
                        help='Defaults to policy_interface.json beside the checkpoint')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(4)
    interface_file = args.interface or args.checkpoint.parent / 'policy_interface.json'
    interface = json.loads(interface_file.read_text())
    manifest_file = args.checkpoint.parent / 'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        for path in (args.checkpoint, interface_file):
            if sha256(path) != manifest['files'][path.name]['sha256']:
                raise ValueError(f'Bundle checksum mismatch: {path.name}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    motion_sha = sha256(args.motion)
    if motion_sha != checkpoint['provenance']['motion_sha256'] or motion_sha != interface['motion_sha256']:
        raise ValueError('Checkpoint/reference/interface mismatch')
    state = checkpoint['modules']['actor']
    if not torch.equal(torch.tensor(interface['HOME']), state['default_q']):
        raise ValueError('Checkpoint/interface HOME mismatch')
    # Actor.load_state_dict restores the exact saved bounds and both normalizers.
    limits = torch.stack((state['lower'] + state['default_q'] - .01,
                          state['upper'] + state['default_q'] + .01), -1)
    actor = Actor(state['default_q'], limits)
    actor.load_state_dict(state, strict=True)
    for key, value in actor.state_dict().items():
        if not torch.equal(value, state[key]):
            raise ValueError(f'Actor state mismatch: {key}')
    with np.load(args.motion, allow_pickle=False) as archive:
        reference_bodies = tuple(archive['body_names'])
    motion = Motion(args.motion, SimpleNamespace(
        joint_names=interface['joint_names'], body_names=reference_bodies), 'cpu')
    # The existing export method only needs these inference fields. It neither
    # constructs a simulator nor restores optimizer, Replay or training RNG.
    context = SimpleNamespace(actor=actor, device='cpu', output=args.output_dir,
                              env=SimpleNamespace(motion=motion, default=actor.default_q,
                                                  backlash=interface['backlash_encoder_view']))
    args.output_dir.mkdir(parents=True)
    SupertrackingRunner.export(context)
    actor.eval()
    policy = args.output_dir / 'policy.onnx'
    latent_file = args.output_dir / 'motion_latents_50hz.npz'
    checkpoint_sha = sha256(args.checkpoint)
    metadata = {
        'checkpoint': str(checkpoint['iteration']), 'checkpoint_sha256': checkpoint_sha,
        'source_training_checkpoint_sha256': checkpoint.get('bundle', {}).get(
            'source_checkpoint_sha256', checkpoint_sha),
        'task_id': 'Mjlab-Supertracking-Flat-OpenChicken',
        'motion_sha256': motion_sha, 'policy_schema': interface['schema'],
        'joint_names': ','.join(interface['joint_names']),
        'default_joint_pos_exact': json.dumps(interface['HOME']),
        'action_units': 'radians relative to HOME', 'control_frequency_hz': '50',
        'normalization': 'History RunningNorm in ONNX; goal normalization in cached latents',
        'runtime_61D_compatible': 'false', 'motion_latents_file': latent_file.name,
    }
    model = onnx.load(policy)
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, policy)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(policy), sess_options=options, providers=['CPUExecutionProvider'])
    future = motion.future(torch.arange(len(motion.state)))
    with np.load(latent_file) as archive:
        latents = archive['latent'].copy()
    np.testing.assert_array_equal(latents, actor.encode(future).numpy())
    checks = []
    generator = torch.Generator().manual_seed(20261105)
    for batch, scale in [(1, 0.), (2, 1.), (32, 3.), (len(future), 1.), (32, 100.)]:
        history = actor.obs_norm.mean + torch.randn(batch, 10, 48, generator=generator) * actor.obs_norm.var.clamp_min(1e-4).sqrt() * scale
        expected = actor(history, future[:batch]).numpy()
        actual = session.run(None, {'history': history.numpy(), 'motion_latent': latents[:batch]})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        checks.append({'batch': batch, 'normalized_input_scale': scale,
                       'max_abs_error': float(np.max(np.abs(actual - expected)))})
    interface.update(checkpoint=str(checkpoint['iteration']), checkpoint_sha256=checkpoint_sha,
                     onnx_file=policy.name, motion_latents_file=latent_file.name,
                     control_frequency_hz=50, history_order='oldest to newest',
                     motion_latent_shape=list(latents.shape))
    (args.output_dir / 'policy_interface.json').write_text(json.dumps(interface, indent=2) + '\n')
    validation = dict(iteration=checkpoint['iteration'], checkpoint_sha256=checkpoint_sha,
                      exporter='SupertrackingRunner.export (CPU, inference fields only)',
                      onnx_checker='passed', exact_actor_state_match=True,
                      latent_matches_checkpoint_encoder=True, inference_checks=checks, files={})
    for path in (policy, latent_file, args.output_dir / 'policy_interface.json'):
        validation['files'][path.name] = {'size': path.stat().st_size, 'sha256': sha256(path)}
    (args.output_dir / 'export_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
