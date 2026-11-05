"""Restore and verify a complete checkpoint from ordered Git LFS parts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, default=Path('checkpoints/supertracking_v12_30000'))
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / 'manifest.json').read_text())
    if manifest['schema'] != 'supertracking_full_checkpoint_parts_v1':
        raise ValueError('Unsupported checkpoint manifest')
    target = args.bundle_dir / manifest['checkpoint_file']
    if target.exists():
        with target.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != manifest['checkpoint_sha256']:
            raise ValueError(f'Existing checkpoint checksum mismatch: {target}')
        print(f'Already verified: {target}')
        return
    for item in manifest['parts']:
        part = args.bundle_dir / item['file']
        if not part.exists() or part.stat().st_size != item['bytes']:
            raise ValueError(f'Missing/incomplete LFS part: {part}; run git lfs pull')
    temporary = target.with_suffix('.pt.tmp')
    full = hashlib.sha256()
    try:
        with temporary.open('xb') as destination:
            for item in manifest['parts']:
                part = args.bundle_dir / item['file']
                digest = hashlib.sha256()
                with part.open('rb') as source:
                    while block := source.read(8 * 1024**2):
                        destination.write(block)
                        digest.update(block)
                        full.update(block)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError(f'LFS part checksum mismatch: {part}')
                print(f'Verified: {part.name}', flush=True)
        if temporary.stat().st_size != manifest['checkpoint_bytes'] or full.hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Restored checkpoint checksum mismatch')
        temporary.replace(target)
    except Exception:
        # Leave any pre-existing interrupted file for inspection.
        raise
    print(f'Restored: {target}\nSHA-256: {full.hexdigest()}')


if __name__ == '__main__':
    main()
