#!/usr/bin/env python3
"""Recompute every shared-budget cell and native replay before aggregating backbones."""
import argparse
import json
from pathlib import Path
import time

from scripts.run_matched_backbone_source_audit import read_rows, write
from skillbridge.matched_backbone_audit import audit_cases, reference_path, report_run, validate_reference
from skillbridge.original_core_audit import full_replay, score_answer, summarize, validate_resume


def read(path):
    return json.loads(path.read_text())


def collect_run(root, backbone, capability, seed):
    from skillbridge.closed_answer_audit import natural_pairs
    from skillbridge.source_image_controls import controlled_records, image_assignment
    from skillbridge.source_measurement import TEMPLATES
    folder = root/f'runs/mechanism_controls/matched_backbone_source_audit_v1/{backbone}/{capability}/seed-{seed}'
    design = read(folder/'design.json')
    records = read_rows(root/f'data/route_transfer_oral_v2/capabilities/{capability}/understanding/test/records.jsonl')
    plan = audit_cases(backbone, seed)
    core = reference_path(root, backbone, capability, seed)
    validate_reference(design['checkpoint'], design['checkpoint_metadata'], backbone, capability, seed)
    if design['checkpoint'] != str(core) or design['records'] != records or design['cases'] != plan or design['templates'] != TEMPLATES:
        raise ValueError('locked inputs or case plan changed')
    assignment, pairs = image_assignment(records), natural_pairs(records)
    if design['image_permutation'] != assignment or design['natural_pairs'] != pairs:
        raise ValueError('input-only pairing or permutation changed')
    controls = dict(original=records)
    for mode in ('permuted', 'neutral'):
        controls[mode] = controlled_records(records, mode, assignment, folder/'neutral-images')
    cells, replays = [], {}
    for case in plan:
        state, mode, instruction = (case[k] for k in ('state', 'image_control', 'instruction'))
        suffix = '' if instruction=='original' else TEMPLATES[instruction][capability]
        actual = [{**r, 'user_prompt':r['user_prompt']+suffix} for r in controls[mode]]
        path = folder/case['protocol']/mode/instruction/state
        if read(path/'inputs.json') != dict(records=actual):
            raise ValueError('cell inputs differ from the fixed design')
        rows = read_rows(path/'predictions.jsonl')
        if len(rows) != len(records):
            raise ValueError('incomplete cell')
        validate_resume(actual, rows, dict(case, backbone=backbone), seed)
        if any(r['capability']!=capability or r['original_image']!=original['user_image'] for r,original in zip(rows,records)):
            raise ValueError('backbone, capability, or original image identity differs')
        scored = [{**r, **score_answer(r['prediction'], r['reference'], capability)} for r in rows]
        value = summarize(scored, pairs if mode=='original' else [])
        if value != read(path/'summary.json'):
            raise ValueError('saved summary differs from raw responses')
        cells.append(dict(case, **value))
        if case['protocol']=='native_replay' or (backbone=='blip3o' and mode=='original' and instruction=='original'):
            old = read_rows(core.parent/f'endpoints/base/understanding/{state}/predictions.jsonl')
            replay = full_replay(rows, old)
            if not replay['passed']:
                raise ValueError('full native replay differs')
            if case['protocol']=='native_replay':
                replays[state] = replay
    report = dict(report_run(cells, replays, backbone, seed), capability=capability)
    if report != read(folder/'report.json'):
        raise ValueError('per-run report differs from independently recomputed cells')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    campaign = root/'runs/mechanism_controls/matched_backbone_source_audit_v1'
    started = time.time()
    def status(state, **kw):
        write(campaign/'collection-status.json', dict(status=state, updated=time.time(), started=started,
              final_report_current=state=='validated', **kw))
    status('checking_completeness')
    jobs = [(b,c,s) for b in ('blip3o','janusflow') for c in ('counting','spatial') for s in (17,29,43,71)]
    for b,c,s in jobs:
        folder = campaign/b/c/f'seed-{s}'
        if not (folder/'complete.json').exists() or not (folder/'status.json').exists() or read(folder/'status.json').get('status')!='complete':
            status('incomplete'); print('Incomplete: final report withheld.'); return 2
    try:
        status('validating')
        runs = [collect_run(root, *job) for job in jobs]
        aggregates = {}
        for b in ('blip3o', 'janusflow'):
            aggregates[b] = {}
            for c in ('counting', 'spatial'):
                bank = [r for r in runs if r['backbone']==b and r['capability']==c]
                means = {}
                for cell in bank[0]['shared_contrasts']:
                    means[cell] = {}
                    for metric in bank[0]['shared_contrasts'][cell]:
                        values = [r['shared_contrasts'][cell][metric] for r in bank]
                        means[cell][metric] = sum(values)/4 if all(v is not None for v in values) else None
                image_effects = {i:{control:{metric:means['original/'+i][metric]-means[control+'/'+i][metric]
                    for metric in ('legacy_accuracy_delta_pp','unambiguous_correct_rate_delta_pp')}
                    for control in ('permuted','neutral')} for i in ('original','answer_only','answer_only_rephrased')}
                aggregates[b][c] = dict(mean_within_run_effects=means, image_dependence_difference_of_differences_pp=image_effects)
        write(campaign/'final-report.json', dict(complete=True, collection_started=started, runs=runs, aggregates=aggregates,
              raw_summaries_recomputed=True, scope='Diagnostic remeasurement of previously examined tests; shared response budget does not equalize architectures or native input processing',
              inference_unit='Four training runs per backbone and capability; test questions reused across runs'))
        status('validated'); print('All 320 cells and native replays validated.'); return 0
    except Exception as error:
        status('failed', error=str(error)); raise


if __name__ == '__main__':
    raise SystemExit(main())
