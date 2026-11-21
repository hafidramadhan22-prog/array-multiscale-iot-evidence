"""Resolve historical path records against this extracted package, never the old host."""
from pathlib import Path, PurePosixPath


def recorded_run_name(recorded):
    path = PurePosixPath(str(recorded).replace('\\', '/'))
    name = path.name
    if not name or name in {'.', '..'} or ':' in name:
        raise ValueError('Invalid recorded run name')
    return name


def local_run(recorded, workspace):
    base = (Path(workspace) / 'runs').resolve()
    path = (base / recorded_run_name(recorded)).resolve()
    if path.parent != base or not path.is_dir():
        raise ValueError('Recorded run is absent from this package')
    return path


def local_artifact(recorded, workspace, allowed_runs):
    parts = PurePosixPath(str(recorded).replace('\\', '/')).parts
    if '..' in parts or 'runs' not in parts:
        raise ValueError('Invalid recorded artifact path')
    marker = max(i for i, name in enumerate(parts) if name == 'runs')
    tail = parts[marker + 1:]
    if len(tail) < 2 or tail[0] not in set(allowed_runs):
        raise ValueError('Artifact is outside the declared package runs')
    base = (Path(workspace) / 'runs' / tail[0]).resolve()
    path = base.joinpath(*tail[1:]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('Artifact is absent or escapes this package run')
    return path
