#!/usr/bin/env python3
"""Evaluate repaired, full-budget UniWorld cores on the fixed reference panels."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import time


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def campaign_plan():
    return dict(jobs=[dict(capability=c, seed=s) for s in (17, 29, 43, 71)
                     for c in ('counting', 'spatial')],
                answer_budgets=[32, 128], generation_images=16000,
                understanding_answers=24576, include_negative_and_zero_effects=True,
                states=['zero', 'full'], gpu=0, inference_seed=8087,
                generation_kwargs=dict(height=512, width=512, diffusion_steps=28,
                                       guidance_scale=3.5, condition_path='joint'),
                preprocessing='Native AutoProcessor; NF4 base and frozen FLUX bnb-4bit',
                inference='Greedy answers: shared 32 and native 128 tokens kept separate',
                scope='Third complete common-core endpoint backbone; no donor replication claim')


def validate_metadata(md, capability, seed):
    expected = dict(backbone='uniworld_v1', capability=capability, seed=seed,
                    method='tied_core', rank=64, timestep_buckets=1,
                    gain_limit=.2 if capability == 'counting' else .1,
                    target_suffixes='q_proj,v_proj')
    if any(md.get(k) != v for k, v in expected.items()):
        raise ValueError('UniWorld checkpoint identity or parameterization differs')


def validate_records(understanding, generation, capability):
    n = 1024 if capability == 'counting' else 512
    if len(understanding) != n or len(generation) != 1000:
        raise ValueError('full reference panel coverage required')
    for panel in (understanding, generation):
        keys = [r['split_key'] for r in panel]
        if any(not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('missing or duplicate input identity')
    if any(r['capability'] != capability for r in understanding):
        raise ValueError('understanding capability differs')
    expected = ({'t2i_compbench_counting': 1000} if capability == 'counting'
                else {'t2i_compbench_2d': 500, 't2i_compbench_3d': 500})
    if Counter(r['benchmark'] for r in generation) != expected:
        raise ValueError('generation benchmark mixture differs')


def infer(root, base, out, capability, seed, u, g):
    import torch
    from scripts.evaluate_cross_backbone_generation import load_uniworld_v1
    from skillbridge.adapter_checkpoint import load_capability_checkpoint
    from skillbridge.cross_backbone_evaluation import generate_fixed_predictions, generate_understanding_predictions
    from skillbridge.decoder_sanity import bypass_layers, same_pixels
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.sidecar import RoutedLowRankLinear
    from skillbridge.source_measurement import audit_pair, validate_completed_predictions

    core = base / 'core.pt'
    ck = torch.load(core, map_location='cpu', weights_only=True)
    validate_metadata(ck['metadata'], capability, seed)
    full = ck['state_dict']
    if not full or any(tuple(v.shape) != (64, 64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('nonfinite or incorrectly shaped core')
    gain = .2 if capability == 'counting' else .1
    adapter, route, kwargs = load_uniworld_v1(root, rank=64, timestep_buckets=1,
        method='tied_core', gain_limit=gain, target_suffixes=('q_proj', 'v_proj'))
    adapter.model.eval()
    load_capability_checkpoint(route, core)
    states = dict(zero={k: torch.zeros_like(v) for k, v in full.items()}, full=full)
    kwargs['condition_path'] = 'joint'
    if kwargs != campaign_plan()['generation_kwargs']:
        raise ValueError('native decoder settings differ from plan')
    lock_json(out/'design.json', dict(core=str(core), checkpoint_metadata=ck['metadata'],
        understanding_records=u, generation_records=g, answer_budgets=[32, 128],
        generation_kwargs=kwargs, inference_seed=8087, source_selection_uses_target=False))
    # Compare the true frozen path and zero-core path before interpreting effects.
    layers = [m for m in route.modules() if isinstance(m, RoutedLowRankLinear)]
    if len(layers) != len(full):
        raise ValueError('route/checkpoint site coverage differs')
    set_core_state(route, states['zero'])
    answer = adapter.understand(Path(u[0]['user_image']), u[0]['user_prompt'], max_new_tokens=32)
    image = adapter.generate(g[0]['user_prompt'], seed=8087, **kwargs)
    with bypass_layers(layers):
        bypass_answer = adapter.understand(Path(u[0]['user_image']), u[0]['user_prompt'], max_new_tokens=32)
        bypass_image = adapter.generate(g[0]['user_prompt'], seed=8087, **kwargs)
    parity = dict(answer_equal=answer == bypass_answer, pixels_equal=same_pixels(image, bypass_image))
    write(out/'zero-bypass-parity.json', parity)
    if not all(parity.values()):
        raise ValueError('zero core and native bypass differ')
    reuse = None
    if seed != 17:
        from scripts.uniworld_frozen_reuse import prepare_reuse, validate_zero_route
        validate_zero_route(adapter, layers)
        reuse = prepare_reuse(base, out, capability, seed, u, g, kwargs)
    answer_reports = {}
    for budget in (32, 128):
        for state in ('zero', 'full'):
            set_core_state(route, states[state])
            dest = out/'understanding'/f'tokens-{budget}'/state
            write(out/'status.json', dict(status='understanding', budget=budget, state=state, updated=time.time()))
            if not (reuse and state == 'zero'):
                generate_understanding_predictions(adapter, u, dest, understanding_kwargs={'max_new_tokens': budget},
                    metadata=dict(condition=state, max_new_tokens=budget), resume=True)
            validate_completed_predictions(u, rows(dest/'predictions.jsonl'))
        answer_reports[str(budget)] = audit_pair(
            rows(out/'understanding'/f'tokens-{budget}'/'zero/predictions.jsonl'),
            rows(out/'understanding'/f'tokens-{budget}'/'full/predictions.jsonl'))
    write(out/'understanding-report.json', answer_reports)
    for state in ('zero', 'full'):
        set_core_state(route, states[state])
        write(out/'status.json', dict(status='generation', state=state, updated=time.time()))
        if not (reuse and state == 'zero'):
            generate_fixed_predictions(adapter, g, out/'generation'/state, seed=8087,
                generation_kwargs=kwargs, metadata=dict(condition=state), resume=True)
    write(out/'inference-complete.json', dict(understanding_answers=len(u)*4, generation_images=2000,
        zero_bypass_parity=parity, frozen_reuse=reuse,
        reused_answers=2*len(u) if reuse else 0, reused_images=len(g) if reuse else 0,
        evaluated_answers=2*len(u) if reuse else 4*len(u),
        generated_images=len(g) if reuse else 2*len(g)))


def run(root, capability, seed):
    from scripts.run_blip3o_readout_followup import targets
    from scripts.audit_reference_generation import audit_panel
    from skillbridge.source_measurement import disjoint_records
    from skillbridge.mechanism_sequence import lock_json
    base = root/f'runs/mechanism_controls/uniworld_core_replication_v1/{capability}/seed-{seed}'
    out = base/'endpoints'
    configuration = json.loads((base/'training-complete.json').read_text())['configuration']
    steps = 8192 if capability == 'counting' else 4096
    expected = dict(backbone='uniworld_v1', capability=capability, seed=seed,
        steps=steps, examples=steps, method='tied_core', rank=64, timestep_buckets=1,
        gain_limit=.2 if capability == 'counting' else .1, preserve_source_order=True,
        shuffle_source_labels=False, optimizer='adamw', learning_rate=.0001, weight_decay=.01)
    if any(configuration.get(k) != v for k, v in expected.items()):
        raise ValueError('completed training recipe differs')
    u = rows(root/f'data/route_transfer_oral_v2/capabilities/{capability}/understanding/test/records.jsonl')
    g_path = targets(root, capability); g = rows(g_path)
    validate_records(u, g, capability)
    disjoint_records(rows(base/'training-inputs.jsonl'), u)
    infer(root, base, out, capability, seed, u, g)
    import torch
    gc.collect(); torch.cuda.empty_cache()
    write(out/'status.json', dict(status='official_scoring', updated=time.time()))
    scorer = root/'environments/t2i_compbench/.venv/bin/python'
    for state in ('zero', 'full'):
        dest = out/'generation'/state; stage = dest/'official-stage'
        if not (stage/'staging_manifest.json').exists():
            subprocess.run([str(root/'environments/uniworld_v1/.venv/bin/python'), '-P',
                str(root/'scripts/stage_oral_predictions.py'), '--records', str(g_path),
                '--predictions', str(dest/'predictions.jsonl'), '--output', str(stage)], check=True)
        summary = stage/'t2i_compbench/official-summary.json'
        if not summary.exists():
            subprocess.run([str(scorer), '-P', str(root/'scripts/run_t2i_compbench_official.py'),
                '--repository', str(root/'external/T2I-CompBench'), '--lock', str(root/'configs/evaluators.lock.json'),
                '--staged-root', str(stage/'t2i_compbench'), '--output', str(summary),
                '--python', str(scorer), '--no-revision-verification'], check=True)
    panels = {s: audit_panel(out/'generation'/s, g, s, 512) for s in ('zero', 'full')}
    lock_json(out/'complete.json', dict(complete=True, capability=capability, seed=seed,
        understanding=json.loads((out/'understanding-report.json').read_text()), generation=panels,
        generation_delta_x100=panels['full']['score_x100']-panels['zero']['score_x100']))
    write(out/'status.json', dict(status='complete', updated=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--capability', choices=('counting', 'spatial'), required=True)
    p.add_argument('--seed', type=int, choices=(17, 29, 43, 71), required=True)
    a = p.parse_args(); root = a.root.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('registered GPU 0 required')
    from scripts.run_uniworld_core_replication import admit_idle_gpu
    from skillbridge.training_overlap import training_lock
    out = root/f'runs/mechanism_controls/uniworld_core_replication_v1/{a.capability}/seed-{a.seed}/endpoints'
    with training_lock(root/'runs/mechanism_controls/gpu_serial_slots/gpu-0', blocking=False):
        try:
            admit_idle_gpu()
            run(root, a.capability, a.seed)
        except Exception as error:
            write(out/'status.json', dict(status='failed', error=str(error), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
