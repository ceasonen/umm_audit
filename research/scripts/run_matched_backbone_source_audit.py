#!/usr/bin/env python3
"""Replay reference cores natively, then infer both backbones under a shared budget."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time

from skillbridge.matched_backbone_audit import audit_cases, predict, reference_path, report_run, validate_reference
from skillbridge.original_core_audit import verify_idle_context_placement, full_replay, score_answer, summarize, validate_resume


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def resume_completed(root,output,args):
    if not (output/'complete.json').exists() or not (output/'status.json').exists():return False
    if json.loads((output/'status.json').read_text()).get('status')!='complete':return False
    from scripts.collect_matched_backbone_source_audit import collect_run
    collect_run(root,args.backbone,args.capability,args.seed)
    write(output/'resume-validation.json',dict(raw_responses_recomputed=True,updated=time.time()))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--backbone', choices=('blip3o', 'janusflow'), required=True)
    p.add_argument('--capability', choices=('counting', 'spatial'), required=True)
    p.add_argument('--seed', type=int, choices=(17, 29, 43, 71), required=True)
    p.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--admission-wait-seconds',type=int,default=43200)
    p.add_argument('--idle-context-pid', type=int, action='append', default=[])
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise ValueError('physical GPU assignment must match CUDA_VISIBLE_DEVICES')
    root = a.root.resolve()
    output = root/f'runs/mechanism_controls/matched_backbone_source_audit_v1/{a.backbone}/{a.capability}/seed-{a.seed}'
    from skillbridge.training_overlap import training_lock
    with ExitStack() as stack:
        stack.enter_context(training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{a.gpu}', blocking=False))
        # An assisting GPU can reach the same registered job. Wait for its
        # writer, then validate completed raw responses before any inference.
        stack.enter_context(training_lock(output, blocking=True))
        if resume_completed(root,output,a):return
        write(output/'status.json', dict(status='checking_admission', updated=time.time(), final_report_current=False))
        try:
            from scripts.run_original_core_source_audit import await_idle_admission
            uid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            def observe():
                free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                    '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'],text=True)
                processes=[]
                for line in lines.splitlines():
                    device,pid,memory=(x.strip() for x in line.split(','))
                    if device==uid:processes.append(dict(pid=int(pid),memory_mib=int(memory)))
                for pid in a.idle_context_pid:
                    if pid not in {v['pid'] for v in processes}:continue
                    argv=[v.decode() for v in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if v]
                    verify_idle_context_placement(argv,a.gpu)
                return free,util,processes
            def waiting(attempt,free,util,processes,error):
                write(output/'status.json',dict(status='waiting_for_idle_gpu',attempt=attempt,gpu=a.gpu,
                    free_mib=free,utilization=util,processes=processes,reason=error,updated=time.time(),final_report_current=False))
            free,util,processes=await_idle_admission(a.gpu,observe,a.idle_context_pid,
                attempts=max(1,a.admission_wait_seconds//5),on_wait=waiting)
            write(output/'gpu_admission.json',dict(gpu=a.gpu,gpu_uuid=uid,free_mib=free,utilization=util,
                processes=processes,allowed_idle_pids=a.idle_context_pid,updated=time.time()))
            evaluate(a, root, output)
        except Exception as error:
            write(output/'status.json', dict(status='failed', error=str(error), updated=time.time(), final_report_current=False))
            raise


def evaluate(a, root, output):
    import torch
    from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o, load_janusflow
    from skillbridge.closed_answer_audit import natural_pairs
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.source_image_controls import controlled_records, image_assignment
    from skillbridge.source_measurement import TEMPLATES
    torch.cuda.set_per_process_memory_fraction(.85, device=0)
    torch.manual_seed(8087)
    core = reference_path(root, a.backbone, a.capability, a.seed)
    checkpoint = torch.load(core, map_location='cpu', weights_only=True)
    sites = validate_reference(core, checkpoint['metadata'], a.backbone, a.capability, a.seed)
    full = checkpoint['state_dict']
    if len(full) != sites or any(tuple(v.shape) != (64, 64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('checkpoint must contain the expected finite rank-64 square cores')
    states = dict(full=full, zero={k:torch.zeros_like(v) for k,v in full.items()})
    records = read_rows(root/f'data/route_transfer_oral_v2/capabilities/{a.capability}/understanding/test/records.jsonl')
    if len(records) != (1024 if a.capability == 'counting' else 512):
        raise ValueError('held-out input panel differs')
    historical = {s:read_rows(core.parent/f'endpoints/base/understanding/{s}/predictions.jsonl') for s in states}
    for rows in historical.values():
        if len(rows) != len(records) or any((r['split_key'], r['user_image'], r['user_prompt'], r['answer_text']) !=
            (v['split_key'], v['image'], v['question'], v['reference']) for r,v in zip(records, rows)):
            raise ValueError('reference historical answers use different inputs')
    assignment, pairs = image_assignment(records), natural_pairs(records)
    design = dict(backbone=a.backbone, capability=a.capability, source_seed=a.seed, checkpoint=str(core),
                  checkpoint_bytes=core.stat().st_size, checkpoint_metadata=checkpoint['metadata'], records=records,
                  cases=audit_cases(a.backbone, a.seed), image_permutation=assignment, natural_pairs=pairs,
                  templates=TEMPLATES, decode='native chat and preprocessing; greedy; explicit per-case token budget',
                  scope='Previously examined test records; diagnostic remeasurement, not fresh confirmation',
                  unmatched='Model widths, native precision, preprocessing, chat templates, and number of trainable sites differ',
                  image_controls='Neutral is RGB127 at original dimensions; permuted keeps original question/answer; both are distribution changes',
                  natural_pair_scope='Same question, different image and answer; other image properties differ',
                  inclusion='Report all registered runs regardless of effect sign; no target-score selection')
    lock_json(output/'design.json', design)
    controls = dict(original=records)
    for mode in ('permuted', 'neutral'):
        controls[mode] = controlled_records(records, mode, assignment, output/'neutral-images')
    gain, suffixes = checkpoint_route_spec(core)
    kwargs = dict(rank=64, timestep_buckets=1, method='tied_core', gain_limit=gain, target_suffixes=suffixes)
    write(output/'status.json', dict(status='loading_model', updated=time.time(), backbone=a.backbone, final_report_current=False))
    adapter, route, _ = (load_blip3o(root, task='understanding', **kwargs) if a.backbone=='blip3o' else load_janusflow(root, **kwargs))
    adapter.model.eval()
    started = time.monotonic()
    cells, replays = [], {}
    for case in design['cases']:
        state, control, instruction = (case[k] for k in ('state', 'image_control', 'instruction'))
        set_core_state(route, states[state])
        suffix = '' if instruction=='original' else TEMPLATES[instruction][a.capability]
        actual = [{**r, 'user_prompt':r['user_prompt']+suffix} for r in controls[control]]
        folder = output/case['protocol']/control/instruction/state
        lock_json(folder/'inputs.json', dict(records=actual))
        path = folder/'predictions.jsonl'
        rows = read_rows(path) if path.exists() else []
        done = validate_resume(actual, rows, dict(case, backbone=a.backbone), a.seed)
        # Resume a native replay only if every already-saved answer matches.
        if case['protocol']=='native_replay' and not full_replay(rows, historical[state][:done])['passed']:
            raise RuntimeError('saved native replay differs from history')
        with path.open('a') as handle:
            for i, row in enumerate(actual[done:], start=done):
                prediction = predict(adapter, row, case)
                result = dict(split_key=row['split_key'], image=row['user_image'], question=row['user_prompt'],
                    reference=row['answer_text'], prediction=prediction, capability=a.capability, source_seed=a.seed,
                    backbone=a.backbone, original_image=records[i]['user_image'], **case,
                    **score_answer(prediction, row['answer_text'], a.capability))
                handle.write(json.dumps(result, allow_nan=False)+'\n'); handle.flush(); rows.append(result)
                if case['protocol']=='native_replay' and prediction != historical[state][i]['prediction']:
                    write(output/f'native_replay_{state}.json', dict(passed=False, first_mismatch=i, result=result,
                          historical_prediction=historical[state][i]['prediction']))
                    raise RuntimeError('native answer replay differs; stop before shared-budget interpretation')
                if (i+1)%32==0 or i+1==len(actual):
                    status = dict(status='running', case=case, rows=i+1, total=len(actual), backbone=a.backbone,
                                  elapsed=time.monotonic()-started, updated=time.time(), final_report_current=False)
                    write(output/'status.json', status); print(json.dumps(status), flush=True)
        validate_resume(actual, rows, dict(case, backbone=a.backbone), a.seed)
        if case['protocol']=='native_replay':
            replays[state] = full_replay(rows, historical[state])
            write(output/f'native_replay_{state}.json', replays[state])
        if a.backbone=='blip3o' and control=='original' and instruction=='original' and not full_replay(rows, historical[state])['passed']:
            raise RuntimeError('BLIP3-o shared 32-token original answers must also replay')
        # Reparse raw text on every resume; do not trust saved correctness flags.
        scored = [{**r, **score_answer(r['prediction'], r['reference'], a.capability)} for r in rows]
        value = summarize(scored, pairs if control=='original' else [])
        lock_json(folder/'summary.json', value)
        cells.append(dict(case, **value))
    report = report_run(cells, replays, a.backbone, a.seed)
    write(output/'report.json', dict(report, capability=a.capability))
    write(output/'complete.json', dict(cases=20, rows_per_case=len(records), native_replay_passed=True, updated=time.time()))
    write(output/'status.json', dict(status='complete', final_report_current=True, updated=time.time()))


if __name__ == '__main__':
    main()
