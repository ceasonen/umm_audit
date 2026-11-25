"""Restore complete adapters for fair source-learning measurements, without hashes."""
import json

import torch

from skillbridge.adapter_checkpoint import _adapter_parameters
from skillbridge.training_overlap import _validate_artifacts


def validate_reference(folder, *, shuffled, backbone, capability, seed, method):
    config = json.loads((folder/'training-config.json').read_text())
    if config.get('shuffle_source_labels') is not shuffled:
        raise ValueError('wrong label condition')
    names = ['adapter.pt', 'history.jsonl', 'summary.json', 'training_order.json',
             'training-config.json', 'training-inputs.jsonl']
    if method != 'shared_lora':
        names.append('core.pt')
    _validate_artifacts(folder, config, names)
    checkpoint = torch.load(folder/'adapter.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported adapter format')
    md = checkpoint['metadata']
    if (md['backbone'], md['capability'], md['seed'], md['method']) != (backbone, capability, seed, method):
        raise ValueError('wrong reference identity')
    if any(not torch.isfinite(v).all() for v in checkpoint['state_dict'].values()):
        raise ValueError('nonfinite adapter state')
    if method != 'shared_lora':
        core = torch.load(folder/'core.pt', map_location='cpu', weights_only=True)
        expected = {k: v for k, v in checkpoint['state_dict'].items() if k.endswith('capability_delta')}
        if core['metadata'] != md or set(expected) != set(core['state_dict']) or any(
                not torch.equal(v, core['state_dict'][k]) for k, v in expected.items()):
            raise ValueError('saved core and complete adapter disagree')
    return config, checkpoint


def apply_state(route, state, *, method, zero):
    if method not in ('shared_lora', 'tied_core', 'svd_core', 'additive_tied_core'):
        raise ValueError('unsupported reference operator')
    parameters = _adapter_parameters(route)
    if set(parameters) != set(state):
        raise ValueError('adapter coordinate sets differ')
    for name, value in state.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError('invalid saved adapter tensor')
        if ('.private_output.' in name or name.startswith('private_output.')) and torch.count_nonzero(value):
            raise ValueError('unexpected trained private path in a source-only shared reference')
        if name.endswith('shared_core.weight') and torch.count_nonzero(value):
            raise ValueError('nonzero calibrated core; zero would not equal frozen')
    targets = {name for name in parameters if (name.endswith('shared_output.weight') if method == 'shared_lora'
                                               else name.endswith('capability_delta'))}
    if zero and not targets:
        raise ValueError('no removable trained state')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
            if zero and name in targets:
                parameter.zero_()
