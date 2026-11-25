#!/usr/bin/env python3
"""Re-evaluate original cores, retaining free responses and paired image controls."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time

from skillbridge.original_core_audit import admit_gpu, verify_idle_context_placement, cases, full_replay, score_answer, summarize, validate_checkpoint, validate_resume


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def await_idle_admission(gpu,observe,allowed_idle_pids,*,attempts=120,pause=5,sleep=time.sleep,on_wait=None):
    """Wait for the existing admission rule; never weaken its resource limits."""
    for attempt in range(attempts):
        free,util,processes=observe()
        try:
            admit_gpu(gpu,free,util,processes,allowed_idle_pids=allowed_idle_pids)
            return free,util,processes
        except ValueError as error:
            if on_wait is not None:on_wait(attempt,free,util,processes,str(error))
            if attempt+1==attempts:raise
            sleep(pause)
    raise ValueError('positive admission attempt count required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(17,29,43,71),required=True)
    p.add_argument('--gpu',type=int,choices=range(4),required=True)
    p.add_argument('--capability',choices=('counting','spatial'),required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--admission-wait-seconds',type=int,default=600)
    p.add_argument('--idle-context-pid',type=int,action='append',default=[])
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):
        raise ValueError('physical GPU assignment must match CUDA_VISIBLE_DEVICES')
    root=a.root.resolve()
    output=root/f'runs/mechanism_controls/original_core_source_audit_v1/{a.capability}/seed-{a.seed}'
    from skillbridge.training_overlap import training_lock
    with ExitStack() as stack:
        stack.enter_context(training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{a.gpu}',blocking=False))
        stack.enter_context(training_lock(output,blocking=False))
        gpu_uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        def observe():
            free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            process_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            processes=[]
            for line in process_text.splitlines():
                uid,pid,memory=(x.strip() for x in line.split(','))
                if uid==gpu_uuid:processes.append(dict(pid=int(pid),memory_mib=int(memory)))
            for pid in a.idle_context_pid:
                if pid not in {v['pid'] for v in processes}:continue
                argv=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
                verify_idle_context_placement(argv,a.gpu)
            return free,util,processes
        def waiting(attempt,free,util,processes,error):
            write(output/'status.json',dict(status='waiting_for_idle_gpu',attempt=attempt,gpu=a.gpu,
                free_mib=free,utilization=util,processes=processes,reason=error,updated=time.time()))
        free,util,processes=await_idle_admission(a.gpu,observe,a.idle_context_pid,
            attempts=max(1,a.admission_wait_seconds//5),on_wait=waiting)
        write(output/'gpu_admission.json',dict(gpu=a.gpu,gpu_uuid=gpu_uuid,free_mib=free,utilization=util,
            processes=processes,allowed_idle_pids=a.idle_context_pid,updated=time.time()))
        try:
            evaluate(a,root,output)
        except Exception as error:
            write(output/'status.json',dict(status='failed',error=str(error),updated=time.time()))
            raise


def evaluate(a,root,output):
    import torch
    from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec,load_blip3o
    from skillbridge.closed_answer_audit import natural_pairs
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.source_image_controls import image_assignment,controlled_records
    from skillbridge.source_measurement import TEMPLATES
    torch.cuda.set_per_process_memory_fraction(.85,device=0)
    torch.manual_seed(8087)
    core=root/f'runs/oral_scale_calibrated_v2/tied_core/blip3o/{a.capability}/seed-{a.seed}/core.pt'
    checkpoint=torch.load(core,map_location='cpu',weights_only=True)
    validate_checkpoint(core,checkpoint['metadata'],a.capability,a.seed)
    states={'full':checkpoint['state_dict']}
    if len(states['full'])!=72 or any(tuple(v.shape)!=(64,64) or not torch.isfinite(v).all() for v in states['full'].values()):
        raise ValueError('original core requires 72 finite rank-64 square matrices')
    states['zero']={k:torch.zeros_like(v) for k,v in states['full'].items()}
    prior=root/f'runs/mechanism_controls/blip3o_readout_probe_v1/{a.capability}/seed-{a.seed}/understanding'
    records=read_rows(root/f'data/route_transfer_oral_v2/capabilities/{a.capability}/understanding/test/records.jsonl')
    if len(records)!=(1024 if a.capability=='counting' else 512):
        raise ValueError('original held-out panel changed')
    historical={s:read_rows(prior/s/'predictions.jsonl') for s in states}
    for state,rows in historical.items():
        if len(rows)!=len(records) or any(
            (r['split_key'],r['user_image'],r['user_prompt'],r['answer_text']) !=
            (v['split_key'],v['image'],v['question'],v['reference']) for r,v in zip(records,rows)):
            raise ValueError('historical outputs do not match original test records')
    assignment=image_assignment(records)
    pairs=natural_pairs(records)
    design=dict(backbone='blip3o',capability=a.capability,source_seed=a.seed,
        checkpoint=str(core),checkpoint_bytes=core.stat().st_size,checkpoint_metadata=checkpoint['metadata'],
        records=records,cases=cases(a.seed),image_permutation=assignment,natural_pairs=pairs,
        max_new_tokens=32,templates=TEMPLATES,decode='native adapter.understand; greedy',
        scope='Previously examined original test panel; same-checkpoint diagnostic, not fresh confirmation',
        neutral_image='RGB127 at original dimensions; OOD diagnostic',
        permuted_reference='Original question/answer retained; mismatched-image diagnostic, not donor truth',
        natural_pair_scope='Same question, different real images/answers; other image differences remain confounded',
        original_recipe_recovered=False,selection_uses_model_outputs=False)
    lock_json(output/'design.json',design)
    for state,rows in historical.items():
        scored=[{**v,**score_answer(v['prediction'],v['reference'],a.capability)} for v in rows]
        lock_json(output/'historical'/f'{state}.json',summarize(scored,pairs))
    gain,suffixes=checkpoint_route_spec(core)
    write(output/'status.json',dict(status='loading_model',gpu=a.gpu,seed=a.seed,updated=time.time()))
    adapter,route,_=load_blip3o(root,rank=64,timestep_buckets=1,method='tied_core',
                              gain_limit=gain,target_suffixes=suffixes,task='understanding')
    adapter.model.eval()
    # Do not interpret a new audit as a remeasurement until original answers replay.
    parity=[]
    for state in ('zero','full'):
        set_core_state(route,states[state])
        for i,row in enumerate(records[:16]):
            prediction=adapter.understand(Path(row['user_image']),row['user_prompt'],max_new_tokens=32)
            parity.append(dict(state=state,index=i,expected=historical[state][i]['prediction'],prediction=prediction,
                               exact=prediction==historical[state][i]['prediction']))
    write(output/'parity.json',dict(rows=parity,passed=all(v['exact'] for v in parity)))
    if not all(v['exact'] for v in parity):
        raise RuntimeError('native historical answer replay differs; stop and diagnose before interpreting audit')
    if a.smoke:
        write(output/'status.json',dict(status='smoke_passed',rows=len(parity),updated=time.time()))
        return
    controls={'original':records}
    for mode in ('permuted','neutral'):
        controls[mode]=controlled_records(records,mode,assignment,output/'neutral-images')
    started=time.monotonic()
    for case in design['cases']:
        state,control,instruction=(case[k] for k in ('state','image_control','instruction'))
        set_core_state(route,states[state])
        suffix='' if instruction=='original' else TEMPLATES[instruction][a.capability]
        actual=[{**r,'user_prompt':r['user_prompt']+suffix} for r in controls[control]]
        folder=output/control/instruction/state
        lock_json(folder/'inputs.json',dict(records=actual))
        path=folder/'predictions.jsonl'
        rows=read_rows(path) if path.exists() else []
        done=validate_resume(actual,rows,case,a.seed)
        with path.open('a') as handle:
            for i,row in enumerate(actual[done:],start=done):
                prediction=adapter.understand(Path(row['user_image']),row['user_prompt'],max_new_tokens=32)
                result=dict(split_key=row['split_key'],image=row['user_image'],question=row['user_prompt'],
                    reference=row['answer_text'],prediction=prediction,capability=a.capability,source_seed=a.seed,
                    original_image=records[i]['user_image'],**case,**score_answer(prediction,row['answer_text'],a.capability))
                handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush();rows.append(result)
                if (i+1)%32==0 or i+1==len(actual):
                    progress=dict(status='running',case=case,rows=i+1,total=len(actual),gpu=a.gpu,seed=a.seed,
                                  elapsed=time.monotonic()-started,updated=time.time())
                    write(output/'status.json',progress);print(json.dumps(progress),flush=True)
        validate_resume(actual,rows,case,a.seed)
        if control=='original' and instruction=='original':
            replay=full_replay(rows,historical[state])
            write(output/f'full_replay_{state}.json',replay)
            if not replay['passed']:
                raise RuntimeError('full original-instruction replay differs; diagnose before interpretation')
        # Natural-pair correctness is interpretable only under original images.
        lock_json(folder/'summary.json',summarize(rows,pairs if control=='original' else []))
    write(output/'complete.json',dict(cases=18,rows_per_case=len(records),prefix_parity_passed=True,full_parity_passed=True,
                                    source_acquisition_proven=False,updated=time.time()))
    write(output/'status.json',dict(status='complete',updated=time.time()))


if __name__=='__main__':
    main()
