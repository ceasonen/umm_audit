#!/usr/bin/env python3
"""Serialize reference-backbone diagnostics after the GPU 2 original-core worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_matched_backbone_source_audit import write


def worker_identity(pid):
    directory = Path('/proc')/str(pid)
    try:
        argv = [v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
        started = (directory/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    return dict(argv=argv, process_start_ticks=started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-worker-pid', type=int, required=True)
    p.add_argument('--idle-context-pid', type=int, action='append', default=[])
    a = p.parse_args(); root = a.root.resolve()
    out = root/'runs/mechanism_controls/matched_backbone_source_audit_v1'
    from skillbridge.training_overlap import training_lock
    from skillbridge.mechanism_sequence import lock_json
    with training_lock(out/'queue-lock', blocking=False):
        def status(state, **kw):
            write(out/'queue.json', dict(status=state, updated=time.time(), gpu=2, **kw))
        predecessor = worker_identity(a.source_worker_pid)
        if predecessor is not None:
            argv = predecessor['argv']
            if len(argv)<3 or argv[-2:] != ['2', '29'] or not argv[-3].endswith('/launch_original_core_source_audit.sh'):
                raise ValueError('predecessor is not the registered GPU 2 original-core worker')
        jobs = [dict(backbone=b, capability=cap, seed=seed)
                for seed in (17, 29, 43, 71) for cap in ('counting', 'spatial') for b in ('janusflow', 'blip3o')]
        lock_json(out/'campaign-plan.json', dict(jobs=jobs, gpu=2, shared_max_new_tokens=32,
            native_max_new_tokens=dict(blip3o=32, janusflow=128), cells_per_job=20, planned_answers=245760,
            inclusion='All registered checkpoints; no selection by accuracy or generation scores',
            success='All native replays and shared cells complete and validated, regardless of effect sign'))
        status('waiting_for_source_audit', source_worker_pid=a.source_worker_pid, predecessor=predecessor)
        until = time.monotonic()+24*3600
        while predecessor is not None:
            current = worker_identity(a.source_worker_pid)
            if current is None or current['process_start_ticks'] != predecessor['process_start_ticks']:
                break
            if current != predecessor:
                status('blocked', reason='predecessor identity changed'); return
            if time.monotonic()>until:
                status('queue_expired', reason='original source worker still active after 24 hours'); return
            time.sleep(30)
        source = root/'runs/mechanism_controls/original_core_source_audit_v1'
        for cap in ('counting', 'spatial'):
            folder = source/cap/'seed-29'
            if not (folder/'complete.json').is_file() or not (folder/'status.json').is_file() or json.loads((folder/'status.json').read_text()).get('status') != 'complete':
                status('blocked', reason='source worker did not complete both registered tasks'); return
        try:
            for index, job in enumerate(jobs):
                status('running', job=job, job_index=index, total_jobs=len(jobs))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHONPATH=f'{root}/src:{root}',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1',
                    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                cmd = [str(root/f'environments/{job["backbone"]}/.venv/bin/python'), '-P',
                    str(root/'scripts/run_matched_backbone_source_audit.py'), '--root', str(root), '--gpu', '2',
                    '--backbone', job['backbone'], '--capability', job['capability'], '--seed', str(job['seed'])]
                for pid in a.idle_context_pid:
                    cmd.extend(['--idle-context-pid', str(pid)])
                result = subprocess.run(cmd, env=env)
                if result.returncode:
                    status('failed', job=job, returncode=result.returncode); return
            result = subprocess.run([str(root/'environments/blip3o/.venv/bin/python'), '-P',
                str(root/'scripts/collect_matched_backbone_source_audit.py'), '--root', str(root)], env=env)
            status('validated' if result.returncode==0 else 'collection_failed', returncode=result.returncode)
        except Exception as error:
            status('failed', error=str(error)); raise


if __name__ == '__main__':
    main()
