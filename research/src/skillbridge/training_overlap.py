"""Bounded look-ahead training without duplicate writes or altered experiments."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time


def make_progress_reporter(path, *, total_steps, every=100):
    """Read-only optimizer progress callback; no tensors, scores or RNG are changed."""
    if total_steps <= 0 or every <= 0:
        raise ValueError('positive training budget and report interval required')
    started=time.monotonic()
    def report(completed_steps, _model):
        if not 1 <= completed_steps <= total_steps:
            raise ValueError('progress step outside training budget')
        if completed_steps not in (1,total_steps) and completed_steps % every:
            return
        elapsed=time.monotonic()-started
        rate=completed_steps/max(elapsed,1e-9)
        value=dict(completed_steps=completed_steps,total_steps=total_steps,
                   elapsed_seconds=elapsed,steps_per_second=rate,
                   eta_seconds=(total_steps-completed_steps)/rate,updated=time.time(),
                   note='current-run throughput estimate; excludes later evaluation and verification')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.partial')
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    return report


@contextmanager
def training_lock(output, *, blocking=True):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_artifacts(output, configuration, names):
    for name in names:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'missing or empty completed artifact: {name}')
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines() if line.strip()]
    steps = configuration['steps']
    if len(history) != steps or any(r['step'] != i or not math.isfinite(r['loss']) for i, r in enumerate(history)):
        raise ValueError('training history is incomplete or nonfinite')
    summary = json.loads((output / 'summary.json').read_text())
    if summary['steps'] != steps or summary['examples'] != configuration['examples']:
        raise ValueError('training summary does not match planned budget')


def mark_training_complete(output, configuration):
    names = ['adapter.pt', 'history.jsonl', 'summary.json', 'training_order.json',
             'training-config.json', 'training-inputs.jsonl']
    if configuration.get('method') in ('tied_core', 'additive_tied_core', 'random_basis_tmc') or (output / 'core.pt').exists():
        names.append('core.pt')
    if json.loads((output / 'training-config.json').read_text()) != configuration:
        raise ValueError('training configuration disagrees with completed run')
    _validate_artifacts(output, configuration, names)
    temporary = output / 'training-complete.partial'
    temporary.write_text(json.dumps({'configuration': configuration, 'artifacts': names,
        'scope': 'training finished; not evaluation or mechanism qualification'}, indent=2) + '\n')
    temporary.replace(output / 'training-complete.json')


def completed_training(output, configuration, inputs_text):
    marker = output / 'training-complete.json'
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved['configuration'] != configuration or json.loads((output / 'training-config.json').read_text()) != configuration:
        raise ValueError('completed training configuration changed')
    if (output / 'training-inputs.jsonl').read_text() != inputs_text:
        raise ValueError('completed training inputs changed')
    _validate_artifacts(output, configuration, saved['artifacts'])
    return True


def can_overlap(gpu, seed, state, *, free_mib, active_pids):
    if gpu not in (0, 1, 2, 3):
        raise ValueError('only physical GPU 0 through 3 is permitted')
    return (seed in (17, 29, 43, 71) and state.get('gpu') == gpu
        and state.get('seed') == seed and state.get('status') == 'running'
        and state.get('job') == 'counting-reference-base-generation'
        and free_mib >= 11500 and len(active_pids) == 1)


def training_command(root, job):
    command = [root / f"environments/{job['backbone']}/.venv/bin/python", '-P',
        root / 'scripts/train_cross_backbone_pilot.py', '--root', root,
        '--preserve-source-order', '--target-suffixes', 'q_proj', 'v_proj']
    for key in ('backbone', 'method', 'source_records', 'capability', 'steps', 'examples', 'rank',
                'timestep_buckets', 'image_size', 'seed', 'gain_limit', 'learning_rate', 'optimizer', 'output'):
        command += ['--' + key.replace('_', '-'), job[key]]
    if job['shuffle_source_labels']:
        command += ['--shuffle-source-labels']
    return list(map(str, command))
