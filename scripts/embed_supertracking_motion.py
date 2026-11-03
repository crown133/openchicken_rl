"""Bake a verified Supertracking ONNX bundle's latent reference into the graph."""
import argparse
import json
from pathlib import Path

import onnx

from mjlab_microduck.supertracking.export import (
    SCHEMA, embed_motion_reference, sha256, validate_embedded_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True,
                        help='Bundle containing policy_interface.json and export_validation.json')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    source, output = args.export_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve previous exports')
    interface = json.loads((source / 'policy_interface.json').read_text())
    validation = json.loads((source / 'export_validation.json').read_text())
    policy = source / interface['onnx_file']
    latent_file = source / interface['motion_latents_file']
    # Refuse a mixed-checkpoint bundle before embedding learned motion features.
    for file in (policy, latent_file, source / 'policy_interface.json'):
        if sha256(file) != validation['files'][file.name]['sha256']:
            raise ValueError(f'Export bundle hash mismatch: {file.name}')
    if not validation.get('latent_matches_checkpoint_encoder'):
        raise ValueError('Source bundle must verify latent/actor checkpoint consistency')
    model = embed_motion_reference(policy, latent_file)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get('checkpoint_sha256') != validation['checkpoint_sha256']:
        raise ValueError('ONNX checkpoint provenance does not match export validation')
    checks = validate_embedded_reference(policy, model, latent_file)
    name = policy.stem + '_embedded.onnx'
    output.mkdir(parents=True)
    destination = output / name
    onnx.save_model(model, str(destination), save_as_external_data=False)
    for key in ('motion_latents_file', 'motion_latent', 'motion_latent_shape'):
        interface.pop(key, None)
    interface.update(
        schema=SCHEMA, onnx_file=name,
        inputs={'history': {'shape': ['batch', 10, 48], 'dtype': 'float32'},
                'time_step': {'shape': ['batch', 1], 'dtype': 'int64'}},
        outputs={'action': {'shape': ['batch', 14], 'dtype': 'float32'}},
        reference_storage='embedded', reference_frames=checks['reference_frames'],
        playback_mode='loop', control_frequency_hz=50,
        time_step_semantics='Frame index, not seconds: max(time_step,0) % reference_frames',
        history_order='oldest to newest, including current observation',
        initialization='Fill all 10 history rows with the first valid observation; previous action is zero HOME-relative, time_step=0',
        stepping='Every 20 ms append the current 48D observation, infer at time_step, then increment time_step',
        restart='Reset time_step to 0 and refill history with the current observation; entry posture must match the trained start',
        normalization='History RunningNorm inside ONNX; goal normalization already included in embedded motion',
        source_onnx_sha256=metadata['source_onnx_sha256'],
        embedded_latent_file_sha256=metadata['embedded_latent_file_sha256'],
    )
    (output / 'policy_interface.json').write_text(json.dumps(interface, indent=2) + '\n')
    checks.update(checkpoint_sha256=validation['checkpoint_sha256'],
                  checkpoint=metadata.get('checkpoint'), source_bundle=str(source),
                  source_onnx_sha256=metadata['source_onnx_sha256'],
                  embedded_latent_file_sha256=metadata['embedded_latent_file_sha256'],
                  onnx_sha256=sha256(destination))
    (output / 'export_validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({'onnx': str(destination), **checks}, indent=2))


if __name__ == '__main__':
    main()
