"""Shared source diagnostics with native replay kept as a separate protocol."""
from pathlib import Path

from skillbridge.original_core_audit import cases, contrasts

BACKBONES = {'blip3o': dict(cohort='reference', sites=72, native_tokens=32),
             'janusflow': dict(cohort='replication', sites=48, native_tokens=128)}


def reference_path(root, backbone, capability, seed):
    if backbone not in BACKBONES or capability not in ('counting', 'spatial') or seed not in (17, 29, 43, 71):
        raise ValueError('unregistered backbone, capability, or training run')
    return Path(root)/f'runs/mechanism_controls/mechanism_sequence_v1/{backbone}/{capability}/seed-{seed}'/BACKBONES[backbone]['cohort']/'core.pt'


def validate_reference(path, metadata, backbone, capability, seed):
    expected = reference_path('/', backbone, capability, seed)
    if tuple(Path(path).parts[-8:]) != tuple(expected.parts[-8:]):
        raise ValueError('checkpoint must belong to the specified reference cohort')
    sites = BACKBONES[backbone]['sites']
    required = dict(backbone=backbone, capability=capability, seed=seed, method='tied_core',
                    rank=64, timestep_buckets=1, gain_limit=.2 if capability=='counting' else .1,
                    target_suffixes='q_proj,v_proj', trainable_parameters=sites*64*64,
                    shuffle_source_labels=False)
    if any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError('reference checkpoint identity or parameterization differs')
    return sites


def audit_cases(backbone, seed):
    native = [dict(state=s, image_control='original', instruction='original',
                   protocol='native_replay', max_new_tokens=BACKBONES[backbone]['native_tokens'])
              for s in ('zero', 'full')]
    return native + [dict(c, protocol='shared_budget', max_new_tokens=32) for c in cases(seed)]


def predict(adapter, row, case):
    # Ask the model to decode under this budget. Never truncate an older answer.
    return adapter.understand(Path(row['user_image']), row['user_prompt'], max_new_tokens=case['max_new_tokens'])


def report_run(cells, replays, backbone, seed):
    keys = ('protocol', 'max_new_tokens', 'image_control', 'instruction', 'state')
    expected = {tuple(c[k] for k in keys) for c in audit_cases(backbone, seed)}
    if len(cells) != 20 or {tuple(c[k] for k in keys) for c in cells} != expected:
        raise ValueError('all native and shared-budget cells required')
    if set(replays) != {'zero', 'full'} or not all(r.get('passed') for r in replays.values()):
        raise ValueError('both full native replays must pass')
    counts = {c['rows'] for c in cells}
    if len(counts) != 1 or next(iter(counts)) <= 0 or any(r.get('rows') not in counts for r in replays.values()):
        raise ValueError('native replay and shared panels must cover the same inputs')
    return dict(backbone=backbone, seed=seed, cells=cells, native_replays=replays,
                shared_contrasts=contrasts([c for c in cells if c['protocol']=='shared_budget']),
                scope='Shared token limit, questions, controls, and parser; native chat, preprocessing, precision, width, and site count differ.')
