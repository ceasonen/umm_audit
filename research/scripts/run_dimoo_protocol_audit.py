#!/usr/bin/env python3
"""Audit the existing spatial DiMOO core without changing training or benchmark scores."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import subprocess
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def balanced_prefix(records):
    selected = []
    counts = Counter()
    for index, record in enumerate(records):
        answer = record['answer_text'].lower()
        if answer not in ('yes', 'no'):
            raise ValueError('expected binary source labels')
        if counts[answer] < 2:
            selected.append((index, record)); counts[answer] += 1
        if counts == {'yes': 2, 'no': 2}:
            return selected
    raise ValueError('not enough examples per label')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=range(4), required=True)
    a = p.parse_args(); root = a.root.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise ValueError('physical GPU assignment differs')
    out = root/'runs/mechanism_controls/dimoo_protocol_audit_v1'
    from skillbridge.training_overlap import training_lock
    from skillbridge.original_core_audit import admit_gpu
    with ExitStack() as stack:
        stack.enter_context(training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{a.gpu}', blocking=False))
        stack.enter_context(training_lock(out, blocking=False))
        try:
            uid, free, util = [s.strip() for s in subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(',')]
            processes = []
            for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                                '--format=csv,noheader,nounits'], text=True).splitlines():
                gpu, pid, mem = [s.strip() for s in line.split(',')]
                if gpu == uid: processes.append(dict(pid=int(pid), memory_mib=int(mem)))
            admit_gpu(a.gpu, int(free), int(util), processes, allowed_idle_pids=[])
            write(out/'gpu_admission.json', dict(gpu=a.gpu, free_mib=int(free), utilization=int(util), processes=processes))
            run(root, out)
        except Exception as error:
            write(out/'status.json', dict(status='failed', error=str(error), updated=time.time()))
            raise


def run(root, out):
    import numpy as np
    import torch
    from scripts.evaluate_cross_backbone_generation import load_dimoo, checkpoint_route_spec
    from skillbridge.dimoo_protocol_audit import partition_loss
    from skillbridge.dimoo_runtime import infer_supervised_mask_ratio
    from skillbridge.evaluation import answers_match, response_is_valid
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.routing import mask_ratio_to_bucket
    from skillbridge.sidecar import sidecar_route
    from skillbridge.source_measurement import inspect_answer, TEMPLATES
    torch.cuda.set_per_process_memory_fraction(.95, 0)
    source = root/'runs/source_acquisition_calibration_v3/dimoo/spatial/beta-0p10'
    core = source/'core.pt'
    checkpoint = torch.load(core, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    for key, expected in dict(backbone='dimoo', capability='spatial', seed=2027, method='tied_core', rank=64).items():
        if metadata.get(key) != expected: raise ValueError(f'checkpoint {key} differs')
    token_paths = list((root/'data/cross_backbone_tokens/spatial/understanding/256/n-4096').glob('seed-2027*/records.jsonl'))
    if len(token_paths) != 1: raise ValueError('ambiguous historical training token cache')
    token_records = rows(token_paths[0]); train = balanced_prefix(token_records)
    development = rows(root/'data/route_transfer_oral_v2/capabilities/spatial/understanding/development/records.jsonl')
    selected = balanced_prefix(development)
    design = dict(core=str(core), metadata=metadata, training_token_records=str(token_paths[0]),
        training_indices=[i for i, _ in train], development_indices=[i for i, _ in selected],
        selection='first two inputs per yes/no label in existing order, independent of predictions',
        states=['zero','full'], instructions=['original','answer_only','answer_only_rephrased'],
        inference=dict(steps=128, generation_length=128, block_length=128, temperature=0.0),
        loss='native masked-label CE partitioned into answer, end and padding; no training changes',
        scope='four-input implementation and measurement diagnostic, not a held-out transfer estimate')
    lock_json(out/'design.json', design)
    # Preserve historical scoring files. Report the current parser separately.
    rescored = []
    for beta in ('0p05','0p10','0p20'):
        for scale in (0, 1):
            path = source.parent/f'beta-{beta}/source-development-endpoints/scale-{scale}/predictions.jsonl'
            records = rows(path)
            audited = [dict(split_key=r['split_key'], reference=r['reference'], prediction=r['prediction'],
                correct=answers_match('spatial', r['reference'], r['prediction']),
                valid=response_is_valid('spatial', r['prediction']), lexical=inspect_answer('spatial', r['prediction'])) for r in records]
            write(out/f'historical/beta-{beta}-scale-{scale}.json', audited)
            rescored.append(dict(beta=beta, scale=scale, n=len(audited),
                accuracy=sum(r['correct'] for r in audited)/len(audited),
                valid=sum(r['valid'] for r in audited)/len(audited),
                answer_only=sum(r['lexical']['answer_only'] for r in audited)/len(audited),
                review=sum(r['lexical']['review_required'] for r in audited)/len(audited)))
    write(out/'historical-rescoring.json', rescored)
    write(out/'status.json', dict(status='loading', updated=time.time()))
    gain, suffixes = checkpoint_route_spec(core)
    adapter, route, _ = load_dimoo(root, rank=64, timestep_buckets=8, method='tied_core', gain_limit=gain, target_suffixes=suffixes)
    adapter.model.eval()
    for parameter in adapter.model.parameters(): parameter.requires_grad_(False)
    train_module = adapter._import('train.train')
    processor = train_module.ItemProcessor(adapter.tokenizer, 1024)
    batches = []
    for index, record in train:
        random.seed(20260923+index); np.random.seed(20260923+index)
        ids, labels = processor.process_item(record, training_mode=True)
        if len(ids) != len(labels): raise ValueError('input-label length mismatch')
        batches.append((index, ids, labels))
    write(out/'training-token-inputs.json', [dict(index=i, input_ids=x, labels=y) for i, x, y in batches])
    state_dict = checkpoint['state_dict']
    losses = []; answers = []; replay = []
    from skillbridge.dimoo_inference import _prompt
    official_prompt = adapter._import('utils.prompt_utils').generate_multimodal_understanding_prompt
    write(out/'prompt-comparison.json', dict(project=_prompt('understanding', 'QUESTION'), official=official_prompt('QUESTION'),
        same=_prompt('understanding', 'QUESTION') == official_prompt('QUESTION'),
        image_preprocessing='project historical path uses original-size VQ tokens; official CLI uses crop plus another newline insertion'))
    with torch.inference_mode():
        for state in ('zero','full'):
            set_core_state(route, {k: torch.zeros_like(v) for k, v in state_dict.items()} if state == 'zero' else state_dict)
            for index, ids, labels in batches:
                write(out/'status.json', dict(status='loss_partition', state=state, index=index, updated=time.time()))
                x = torch.tensor([ids], dtype=torch.long, device='cuda'); y = torch.tensor([labels], dtype=torch.long, device='cuda')
                bucket = mask_ratio_to_bucket(infer_supervised_mask_ratio(x, y), buckets=8)
                with sidecar_route(route, task='understanding', timestep_bucket=bucket):
                    logits = adapter.model(input_ids=x, infer=True).logits
                report = partition_loss(logits, x, y)
                losses.append(dict(state=state, index=index, losses=report))
                write(out/'loss-partitions.json', losses)
                del logits, x, y
            historical = rows(source/f'source-development-endpoints/scale-{int(state == "full")}/predictions.jsonl')
            for index, record in selected:
                for instruction in ('original','answer_only','answer_only_rephrased'):
                    question = record['user_prompt'] + ('' if instruction == 'original' else TEMPLATES[instruction]['spatial'])
                    write(out/'status.json', dict(status='inference', state=state, index=index, instruction=instruction,
                        completed=len(answers), total=24, updated=time.time()))
                    torch.manual_seed(8087+index)
                    answer = adapter.understand(Path(record['user_image']), question, **design['inference'])
                    row = dict(state=state, index=index, split_key=record['split_key'], image=record['user_image'],
                        reference=record['answer_text'], question=question, instruction=instruction, prediction=answer,
                        correct=answers_match('spatial', record['answer_text'], answer), lexical=inspect_answer('spatial', answer))
                    answers.append(row); write(out/'answers.json', answers)
                    if instruction == 'original':
                        old = historical[index]
                        if old['split_key'] != row['split_key']: raise ValueError('historical input order differs')
                        replay.append(dict(state=state, index=index, equal=answer == old['prediction'], historical_prediction=old['prediction']))
                        write(out/'historical-replay.json', replay)
                    print(json.dumps(dict(state=state, index=index, instruction=instruction, prediction=answer)), flush=True)
    write(out/'complete.json', dict(complete=True, loss_cells=len(losses), answer_cells=len(answers),
        historical_replay_matches=sum(r['equal'] for r in replay), historical_replay_total=len(replay),
        transfer_evidence=False, scope=design['scope']))
    write(out/'status.json', dict(status='complete', updated=time.time()))


if __name__ == '__main__': main()
