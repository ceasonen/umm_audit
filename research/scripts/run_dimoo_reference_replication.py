#!/usr/bin/env python3
"""Run registered DiMOO technical checks and eight full-budget reference jobs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from skillbridge.dimoo_replication import validate_reference, validate_contract
from skillbridge.mechanism_sequence import lock_json
from skillbridge.training_overlap import training_lock
from scripts.queue_matched_backbone_source_audit import worker_identity
from scripts.run_dimoo_protocol_audit import rows, write


def prepare(root):
    from skillbridge.source_measurement import disjoint_records
    out=root/'runs/mechanism_controls/dimoo_reference_replication_v1';jobs=[]
    for cap in ('spatial','counting'):
        heldout=rows(root/f'data/route_transfer_oral_v2/capabilities/{cap}/understanding/test/records.jsonl')
        for seed in (17,29,43,71):
            reference=root/f'runs/mechanism_controls/mechanism_sequence_v1/blip3o/{cap}/seed-{seed}/reference'
            config=json.loads((reference/'training-config.json').read_text())
            validate_reference(config,cap,seed)
            source=Path(config['source_records']);records=rows(source)
            if records!=rows(reference/'training-inputs.jsonl') or len(records)!=config['steps'] or len({r['split_key'] for r in records})!=len(records):
                raise ValueError('reference source order or coverage differs')
            if any(not Path(r['user_image']).is_file() for r in records):raise ValueError('source image missing')
            disjoint_records(records,heldout)
            dest=out/cap/f'seed-{seed}'
            command=[str(root/'environments/dimoo/.venv/bin/python'),'-u','-P',
                str(root/'scripts/train_dimoo_reference_core.py'),'--root',str(root),
                '--backbone','dimoo','--method','tied_core','--capability',cap,'--seed',str(seed),
                '--source-records',str(source),'--preserve-source-order','--steps',str(config['steps']),
                '--examples',str(config['steps']),'--rank','64','--timestep-buckets','1','--image-size','256',
                '--optimizer','adamw','--learning-rate','0.0001','--gain-limit',str(config['gain_limit']),
                '--target-suffixes','q_proj','v_proj','--optimization-audit','--output',str(dest)]
            jobs.append(dict(capability=cap,seed=seed,steps=config['steps'],output=str(dest),command=command,
                             reference=str(reference),source_records=str(source)))
    design=dict(jobs=jobs,total_updates=49152,gpu=1,rank=64,timestep_buckets=1,
        preprocessing='official 256-scale variable crop; Python/NumPy crop and mask seed equals training run',
        objective='unaltered native masked-label CE, including native padding supervision; full input/label tensors retained',
        training_precision='bfloat16; whole-layer activation checkpointing',
        comparison='same ordered sources, rank, updates, optimizer and gain; native widths/sites/objective differ',
        technical_checks='32 training updates for each capability; actual annotation reconstruction and positive gradients/native parameter updates',
        inclusion='all four runs per capability; no selection by accuracy, loss improvement, or generation scores',
        evaluation_plan=dict(understanding_states=['zero','full'],answer_lengths=[32,128],
            decoding='fixed official system prompt and 256-scale crop, one image wrapper, corrected greedy single-block scheduler',
            scoring='raw span and protocol-delimited answer both retained; benchmark and lexical scores kept separate',
            generation='native DiMOO generation protocol; 1000 prompts per capability and original paired noise',
            status='endpoint implementation and execution remain separate'))
    lock_json(out/'design.json',design)
    return out,jobs


def validate_training(job):
    import torch
    base=Path(job['output']);steps=job['steps']
    if not (base/'training-complete.json').exists():return False
    cfg=json.loads((base/'training-complete.json').read_text())['configuration']
    expected=dict(backbone='dimoo',method='tied_core',capability=job['capability'],seed=job['seed'],
        steps=steps,examples=steps,rank=64,timestep_buckets=1,image_size=256,optimizer='adamw',
        learning_rate=.0001,weight_decay=.01,gain_limit=.2 if job['capability']=='counting' else .1,
        preserve_source_order=True,shuffle_source_labels=False,target_suffixes=['q_proj','v_proj'],
        source_records=job['source_records'])
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('completed recipe differs')
    if rows(base/'training-inputs.jsonl')!=rows(Path(job['source_records']))[:steps]:raise ValueError('completed source order differs')
    h=rows(base/'history.jsonl');t=rows(base/'optimization.jsonl')
    if [r['step'] for r in h]!=list(range(steps)) or any(not math.isfinite(r['loss']) for r in h):
        raise ValueError('training history incomplete or nonfinite')
    if [r['step'] for r in t]!=[1,*range(100,steps,100),steps] or any(
            r['missing_gradient_names'] or not math.isfinite(r['gradient_l2']) or not math.isfinite(r['update_l2']) or
            r['gradient_l2']<=0 or r['update_l2']<=0 for r in t):raise ValueError('native gradient/update check failed')
    ck=torch.load(base/'core.pt',map_location='cpu',weights_only=True);md=ck['metadata']
    for k in ('backbone','method','capability','seed','rank','timestep_buckets','image_size','gain_limit'):
        if md.get(k)!=expected[k]:raise ValueError('checkpoint identity differs')
    if not ck['state_dict'] or any(tuple(v.shape)!=(64,64) or not torch.isfinite(v).all() for v in ck['state_dict'].values()):
        raise ValueError('checkpoint tensor audit failed')
    batches=torch.load(base/'native-training-batches.pt',map_location='cpu',weights_only=True)
    audit=json.loads((base/'native-supervision-audit.json').read_text())
    if len(batches)!=steps or audit['records']!=steps or [r['split_key'] for r in batches]!=[r['split_key'] for r in rows(base/'training-inputs.jsonl')]:
        raise ValueError('native batch archive coverage differs')
    return True


def supplemental_gpu_admissible(gpu, free_mib, utilization, processes):
    return gpu in (3, 6, 7) and free_mib >= 20000 and utilization <= 5 and not processes


def idle_gpu(out, gpu=1):
    from scripts.run_original_core_source_audit import await_idle_admission
    def observe():
        uid,free,util=[v.strip() for v in subprocess.check_output(['nvidia-smi','-i',str(gpu),
            '--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
        processes=[]
        for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                            '--format=csv,noheader,nounits'],text=True).splitlines():
            device,pid,memory=[v.strip() for v in line.split(',')]
            if device==uid:processes.append(dict(pid=int(pid),memory_mib=int(memory)))
        return int(free),int(util),processes
    if gpu == 1:
        await_idle_admission(gpu,observe,[],attempts=8640,on_wait=lambda attempt,free,util,processes,error:
            write(out/'status.json',dict(status='waiting_for_idle_gpu',attempt=attempt,free_mib=free,
                  utilization=util,processes=processes,updated=time.time())))
        return
    if gpu not in (3, 6, 7):
        raise ValueError('supplemental GPU is not authorized for DiMOO')
    for attempt in range(8640):
        free,util,processes=observe()
        if supplemental_gpu_admissible(gpu,free,util,processes):
            return
        write(out/'status.json',dict(status='waiting_for_idle_gpu',attempt=attempt,free_mib=free,
              utilization=util,processes=processes,updated=time.time()))
        time.sleep(5)
    raise ValueError('supplemental GPU did not meet the original idle resource limits')


def run_job(root,out,job):
    if validate_training(job):return
    with training_lock(root/'runs/mechanism_controls/gpu_serial_slots/gpu-1',blocking=False):
        idle_gpu(out);base=Path(job['output']);base.mkdir(parents=True,exist_ok=True)
        start=time.time()
        with (base/'run.log').open('a') as log:
            child=subprocess.Popen(job['command'],cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            write(out/'status.json',dict(status='training',capability=job['capability'],seed=job['seed'],
                steps=job['steps'],child_pid=child.pid,updated=start));code=child.wait()
        if code or not validate_training(job):raise RuntimeError(f'DiMOO training failed: exit {code}')
        write(base/'validated-training.json',dict(complete=True,elapsed_seconds=time.time()-start,steps=job['steps']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--donor-queue-pid',type=int)
    a=p.parse_args();root=a.root.resolve();out,jobs=prepare(root)
    if a.preflight_only:
        write(out/'preflight.json',dict(complete=True,jobs=8,updates=49152,source_order_and_heldout_separation=True));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('registered GPU 1 required')
    with training_lock(out/'queue-lock',blocking=False):
        try:
            if a.donor_queue_pid is not None:
                previous=worker_identity(a.donor_queue_pid)
                if previous is None:raise ValueError('registered predecessor must be live when queue is created')
                argv=previous['argv'];script=str(root/'scripts/run_janusflow_donor_replication.py')
                if script not in argv or '--validation-queue-pid' not in argv or '--root' not in argv or argv[argv.index('--root')+1]!=str(root):
                    raise ValueError('unexpected GPU 1 predecessor')
                write(out/'queue.json',dict(status='waiting_for_donor_queue',pid=a.donor_queue_pid,
                    predecessor=previous,updated=time.time()))
                while True:
                    now=worker_identity(a.donor_queue_pid)
                    if now is None or now['process_start_ticks']!=previous['process_start_ticks']:break
                    if now!=previous:raise ValueError('predecessor identity changed')
                    time.sleep(30)
            contract=root/'runs/mechanism_controls/dimoo_understanding_contract_v1/complete.json'
            validate_contract(json.loads(contract.read_text()))
            # Only technical protocol validity gates training; donor outcomes do not.
            for cap in ('spatial','counting'):
                original=next(j for j in jobs if j['capability']==cap and j['seed']==17)
                smoke={**original,'steps':32,'output':str(out/'technical-training'/cap)}
                command=list(original['command'])
                for key,value in [('--steps','32'),('--examples','32'),('--output',smoke['output'])]:
                    command[command.index(key)+1]=value
                smoke['command']=command;run_job(root,out,smoke)
            write(out/'technical-training-complete.json',dict(complete=True,capabilities=2,updates=64))
            for job in jobs:run_job(root,out,job)
            write(out/'training-complete.json',dict(complete=True,jobs=8,updates=49152,endpoints_complete=False))
            write(out/'status.json',dict(status='training_complete',updated=time.time()))
        except Exception as error:
            write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()));raise


if __name__=='__main__':main()
