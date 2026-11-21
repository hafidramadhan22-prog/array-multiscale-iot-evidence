"""Verify every delivered evidence file without loading a model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text('utf8'))
    for row in manifest['files']:
        p = (ROOT / row['path']).resolve()
        if not p.is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe package path')
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Package mismatch: ' + row['path'])
    print(json.dumps(dict(status='PASS_ALL_PACKAGE_FILES',files=len(manifest['files']),
                          total_bytes=sum(r['bytes'] for r in manifest['files'])), indent=2))


if __name__ == '__main__':
    main()
