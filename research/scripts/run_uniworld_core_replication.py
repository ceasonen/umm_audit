#!/usr/bin/env python3
"""Full-budget UniWorld training with the reference recipe and repaired labels."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from skillbridge.mechanism_sequence import lock_json
from skillbridge.original_core_audit import admit_gpu
from skillbridge.training_overlap import training_lock
from scripts.run_dimoo_protocol_audit import write


def admit_idle_gpu(*, attempts=120, pause=5):
    """Wait for process release/utilization sampling without relaxing admission."""
    for attempt in range(attempts):
        uid, free, util = [v.strip() for v in subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')]
        processes = []
        for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                            '--format=csv,noheader,nounits'], text=True).splitlines():
            gpu, pid, mem = [v.strip() for v in line.split(',')]
            if gpu == uid: processes.append(dict(pid=int(pid), memory_mib=int(mem)))
        try:
            admit_gpu(0, int(free), int(util), processes, allowed_idle_pids=[])
            return
        except ValueError:
            if attempt + 1 == attempts: raise
            time.sleep(pause)


def completed_job(job):
    """Resume from complete artifacts, never from a stale progress counter."""
    base = Path(job['output']); marker = base/'training-complete.json'
    if not marker.exists(): return False
    cfg = json.loads(marker.read_text())['configuration']
    steps = 8192 if job['capability'] == 'counting' else 4096
    source = job['command'][job['command'].index('--source-records')+1]
    expected = dict(backbone='uniworld_v1', capability=job['capability'], seed=job['seed'],
        method='tied_core', rank=64, timestep_buckets=1, steps=steps, examples=steps,
        optimizer='adamw', learning_rate=.0001, weight_decay=.01,
        gain_limit=.2 if job['capability']=='counting' else .1,
        target_suffixes=['q_proj','v_proj'], preserve_source_order=True,
        shuffle_source_labels=False, source_records=source)
    if any(cfg.get(k) != v for k,v in expected.items()):
        raise ValueError('completed job recipe differs')
    if any(not (base/name).is_file() or not (base/name).stat().st_size for name in ('core.pt','adapter.pt')):
        raise ValueError('completed checkpoint artifact missing')
    history = [json.loads(s) for s in (base/'history.jsonl').read_text().splitlines() if s.strip()]
    telemetry = [json.loads(s) for s in (base/'optimization.jsonl').read_text().splitlines() if s.strip()]
    if len(history) != steps or [r.get('step') for r in history] != list(range(steps)) or any(not math.isfinite(r['loss']) for r in history):
        raise ValueError('completed training history is incomplete or nonfinite')
    if [r['step'] for r in telemetry] != [1, *range(100, steps, 100), steps] or any(
            r['missing_gradient_names'] or not r['gradient_l2']>0 or not r['update_l2']>0 for r in telemetry):
        raise ValueError('completed gradient/update telemetry failed')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve();out=root/'runs/mechanism_controls/uniworld_core_replication_v1'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('registered GPU 0 required')
    with training_lock(out/'worker',blocking=False):
        try:
            smoke=json.loads((root/'runs/mechanism_controls/uniworld_label_repair_smoke_v1/complete.json').read_text())
            if not smoke['complete'] or not smoke['gradient_update_checks_passed']:raise ValueError('technical training check failed')
            jobs=[]
            for cap,steps,gain in (('spatial',4096,.1),('counting',8192,.2)):
                for seed in (17,29,43,71):
                    reference=root/f'runs/mechanism_controls/mechanism_sequence_v1/blip3o/{cap}/seed-{seed}/reference/training-config.json'
                    config=json.loads(reference.read_text())
                    required=dict(backbone='blip3o',capability=cap,method='tied_core',seed=seed,steps=steps,examples=steps,
                        rank=64,timestep_buckets=1,optimizer='adamw',learning_rate=1e-4,weight_decay=.01,gain_limit=gain,
                        target_suffixes=['q_proj','v_proj'],preserve_source_order=True,shuffle_source_labels=False)
                    if any(config.get(k)!=v for k,v in required.items()):raise ValueError(f'reference recipe differs: {reference}')
                    source=Path(config['source_records'])
                    records=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
                    if len(records)!=steps or len({r['split_key'] for r in records})!=steps:raise ValueError('source order coverage differs')
                    dest=out/cap/f'seed-{seed}'
                    command=[str(root/'environments/uniworld_v1/.venv/bin/python'),'-u','-P',str(root/'scripts/train_cross_backbone_pilot.py'),
                        '--root',str(root),'--backbone','uniworld_v1','--method','tied_core','--capability',cap,
                        '--source-records',str(source),'--preserve-source-order','--steps',str(steps),'--examples',str(steps),
                        '--rank','64','--timestep-buckets','1','--image-size',str(config['image_size']),'--seed',str(seed),
                        '--optimizer','adamw','--learning-rate','0.0001','--gain-limit',str(gain),
                        '--target-suffixes','q_proj','v_proj','--optimization-audit','--output',str(dest)]
                    jobs.append(dict(capability=cap,seed=seed,source_reference=str(reference),output=str(dest),command=command))
            lock_json(out/'design.json',dict(backbone='uniworld_v1',jobs=jobs,gpu=0,
                label_rule='answer-only labels after expanded multimodal prefix equality check',
                pretrained_weights='UniWorld-V1-NF4',preprocessing='native AutoProcessor; historical image-size argument is unused for UniWorld',
                inclusion='all completed runs, independent of source/target gain',
                scope='full reference-budget training; source and generation endpoints require separate evaluation',
                semantic_acquisition_gate=False))
            for index,job in enumerate(jobs):
                if completed_job(job):
                    print(json.dumps(dict(status='verified_complete',job=index)),flush=True)
                    continue
                with training_lock(root/'runs/mechanism_controls/gpu_serial_slots/gpu-0',blocking=False):
                    write(out/'status.json',dict(status='waiting_for_idle_gpu',job=index,updated=time.time()))
                    admit_idle_gpu()
                    destination=Path(job['output']);destination.mkdir(parents=True,exist_ok=True)
                    with (destination/'run.log').open('a') as log:
                        child=subprocess.Popen(job['command'],cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                        write(out/'status.json',dict(status='training',job=index,total=8,capability=job['capability'],seed=job['seed'],child_pid=child.pid,updated=time.time()))
                        code=child.wait()
                    if code or not (destination/'training-complete.json').exists():raise RuntimeError(f'job {index} failed, exit {code}')
                    if not completed_job(job):raise ValueError('missing verified training completion')
            write(out/'training-complete.json',dict(complete=True,jobs=8,understanding_evaluated=False,generation_evaluated=False))
            write(out/'status.json',dict(status='training_complete',updated=time.time()))
        except Exception as error:
            write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()));raise


if __name__=='__main__':main()
