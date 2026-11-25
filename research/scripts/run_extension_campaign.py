#!/usr/bin/env python3
"""Independent model workers: idle GPU admission, native checks, full training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from skillbridge.training_overlap import training_lock


def gpu_eligible(gpu, authorized, free_mib, utilization, pids):
    return gpu in authorized and free_mib >= 20000 and utilization <= 5 and not pids


def smoke_passed(report):
    return (report.get('complete') is True and report.get('steps') == 32
            and report.get('zero_core_equal') is True
            and report.get('gradient_update_checks_passed') is True)


def campaign_jobs(seeds, *, training_only=False):
    jobs=[] if training_only else [('smoke',cap,17) for cap in ['counting','spatial']]
    return jobs+[('training',cap,seed) for seed in seeds for cap in ['counting','spatial']]


def training_command(root, model, capability, seed):
    steps = 8192 if capability == 'counting' else 4096
    out = root / f'runs/mechanism_controls/{model}_reference_replication_v1/{capability}/seed-{seed}'
    return [str(root / f'environments/{model}/.venv/bin/python'), '-u', '-P',
        str(root / f'scripts/train_{model}_reference_core.py'), '--root', str(root),
        '--backbone', model, '--method', 'tied_core', '--capability', capability,
        '--seed', str(seed), '--source-records',
        str(root / f'runs/mechanism_controls/mechanism_sequence_v1/data/{capability}/seed-{seed}/ordered-source.jsonl'),
        '--preserve-source-order', '--steps', str(steps), '--examples', str(steps),
        '--rank', '64', '--timestep-buckets', '1', '--image-size', '384' if model == 'januspro' else '432',
        '--optimizer', 'adamw', '--learning-rate', '0.0001',
        '--gain-limit', '0.2' if capability == 'counting' else '0.1',
        '--target-suffixes', 'q_proj', 'v_proj', '--optimization-audit', '--output', str(out)]


def idle(gpu, authorized):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uid, free, util = [v.strip() for v in raw.strip().split(',')]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    pids = [line.split(',')[1].strip() for line in apps.splitlines() if line.split(',')[0].strip() == uid]
    return gpu_eligible(gpu, authorized, int(free), int(util), pids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['januspro', 'showo2'], required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--seeds', type=int, nargs='+', default=[17,29,43,71])
    p.add_argument('--training-only', action='store_true')
    p.add_argument('--worker-id', default='campaign')
    args = p.parse_args()
    root = args.root.resolve(); model = args.model
    work = root / f'runs/mechanism_controls/backbone_extension_v1/{model}'
    work.mkdir(parents=True, exist_ok=True)
    def status(**data):
        tmp = work / f'{args.worker_id}-status.partial'
        tmp.write_text(json.dumps(dict(data, updated=time.time(), authorized_gpus=args.gpus), indent=2)+'\n')
        tmp.replace(work / f'{args.worker_id}-status.json')
    jobs = campaign_jobs(args.seeds, training_only=args.training_only)
    finished = []
    with training_lock(work / args.worker_id, blocking=False):
        for stage, cap, seed in jobs:
            smoke = work / f'{cap}-smoke'
            out = smoke if stage == 'smoke' else root / f'runs/mechanism_controls/{model}_reference_replication_v1/{cap}/seed-{seed}'
            marker = out / ('smoke-complete.json' if stage == 'smoke' else 'training-complete.json')
            if marker.exists():
                if stage == 'smoke' and not smoke_passed(json.loads(marker.read_text())):
                    raise ValueError('incomplete native checks')
                if stage == 'smoke':
                    finished.append([stage, cap, seed]); continue
                # Trainer revalidates completed artifacts and the exact recipe.
            if stage == 'training':
                while not (smoke/'smoke-complete.json').exists():
                    status(state='waiting_for_native_checks', capability=cap, seed=seed, finished=finished)
                    time.sleep(15)
                if not smoke_passed(json.loads((smoke/'smoke-complete.json').read_text())):
                    raise ValueError('full training requires passed native checks')
            command = training_command(root, model, cap, seed) if stage == 'training' else [
                str(root / f'environments/{model}/.venv/bin/python'), '-u', '-P',
                str(root / 'scripts/run_extension_training_smoke.py'), '--root', str(root),
                '--model', model, '--capability', cap]
            while True:
                ran = False
                for gpu in args.gpus:
                    if not idle(gpu, args.gpus): continue
                    try:
                        with training_lock(root / f'runs/mechanism_controls/gpu_serial_slots/gpu-{gpu}', blocking=False):
                            if not idle(gpu, args.gpus): continue
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='src:.',
                                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
                            out.mkdir(parents=True, exist_ok=True)
                            with (out / 'campaign-run.log').open('a') as log:
                                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                                status(state='running', stage=stage, capability=cap, seed=seed,
                                       gpu=gpu, pid=child.pid, finished=finished, log=str(out/'campaign-run.log'))
                                code = child.wait()
                            if code != 0 or not marker.exists():
                                status(state='failed', stage=stage, capability=cap, seed=seed,
                                       exit_code=code, finished=finished, log=str(out/'campaign-run.log'))
                                raise SystemExit(1)
                            if stage == 'smoke' and not smoke_passed(json.loads(marker.read_text())):
                                raise ValueError('native checks did not pass')
                            finished.append([stage, cap, seed]); ran = True; break
                    except BlockingIOError:
                        continue
                if ran: break
                status(state='waiting_for_idle_gpu', stage=stage, capability=cap, seed=seed, finished=finished)
                time.sleep(15)
        status(state='training_finished', finished=finished, endpoint_evaluation=False)


if __name__ == '__main__':
    main()
