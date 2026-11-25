#!/usr/bin/env python3
"""Measure fixed -Q endpoints after complete trajectory qualification."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def family_status(qualification,capability):
    if qualification.get('complete') is not True:raise ValueError('complete trajectory campaign required')
    family=qualification['capabilities'][capability];runs=family['runs']
    if len(runs)!=4 or {r['seed'] for r in runs}!={17,29,43,71} or any(type(r['matched']) is not bool for r in runs):
        raise ValueError('all four training runs required; no successful subset')
    matched=all(r['matched'] for r in runs)
    if family['matched'] is not matched:raise ValueError('family qualification inconsistent')
    return matched


def validate_baseline_design(design,*,task,seed,core,records,count):
    expected=dict(task=task,source_seed=seed,core=str(core),records=str(records),rows=count,
                  inference_seed=8087,cache_policy='reset before every generated image')
    if any(design.get(k)!=v for k,v in expected.items()) or not {'zero','full'}<=set(design.get('condition_order',[])):
        raise ValueError('historical baseline protocol differs')


def baseline_evidence(root,capability,seed):
    from scripts.run_blip3o_readout_followup import targets
    from scripts.audit_reference_generation import audit_panel
    from scripts.analyze_readout_evidence import endpoint
    from skillbridge.source_measurement import validate_completed_predictions,audit_rows
    core=root/f'runs/oral_scale_calibrated_v2/tied_core/blip3o/{capability}/seed-{seed}/core.pt'
    baseline=root/f'runs/mechanism_controls/blip3o_readout_probe_v1/{capability}/seed-{seed}'
    upath=root/f'data/route_transfer_oral_v2/capabilities/{capability}/understanding/test/records.jsonl'
    gpath=targets(root,capability);u=rows(upath);g=rows(gpath)
    if len(u)!=(1024 if capability=='counting' else 512) or len(g)!=1000:
        raise ValueError('reference panel size differs')
    for task,path,records in [('understanding',upath,u),('generation',gpath,g)]:
        validate_baseline_design(read(baseline/task/'design.json'),task=task,seed=seed,
                                 core=core,records=path,count=len(records))
    evidence=dict(understanding={},generation={})
    for state in ('zero','full'):
        ur=rows(baseline/'understanding'/state/'predictions.jsonl')
        validate_completed_predictions(u,ur)
        endpoint(baseline/'understanding'/state,'understanding')
        for task in ('understanding','generation'):
            records=rows(baseline/task/state/'predictions.jsonl')
            if any(r.get('source_seed')!=seed or r.get('intervention')!=state for r in records):
                raise ValueError('historical state/run identity differs')
        evidence['understanding'][state]=audit_rows(ur)[0]
        evidence['generation'][state]=audit_panel(baseline/'generation'/state,g,state,512,condition_key='intervention')
    return core,baseline,u,g,gpath,evidence


def infer_task(root,out,core,baseline,capability,seed,task,records):
    import torch
    from PIL import Image
    from scripts.evaluate_cross_backbone_generation import load_blip3o,checkpoint_route_spec
    from scripts.run_blip3o_readout_probe import FreshGeneration
    from skillbridge.cross_backbone_evaluation import generate_fixed_predictions,generate_understanding_predictions
    from skillbridge.decoder_sanity import same_pixels
    from skillbridge.fixed_core_trajectory import negated_state
    from skillbridge.original_core_audit import validate_checkpoint
    from skillbridge.readout_probe import set_core_state
    from skillbridge.source_measurement import validate_completed_predictions
    ck=torch.load(core,map_location='cpu',weights_only=True);validate_checkpoint(core,ck['metadata'],capability,seed)
    full=ck['state_dict']
    if len(full)!=72 or any(tuple(v.shape)!=(64,64) for v in full.values()):raise ValueError('original core sites differ')
    gain,suffixes=checkpoint_route_spec(core)
    adapter,route,kwargs=load_blip3o(root,rank=64,timestep_buckets=1,method='tied_core',
                                  gain_limit=gain,target_suffixes=suffixes,task=task)
    adapter.model.eval();fresh=FreshGeneration(adapter);replay=[]
    if task=='generation' and kwargs!=dict(height=512,width=512,diffusion_steps=30,guidance_scale=3.0):
        raise ValueError('original native generation configuration differs')
    for state,values in [('zero',{k:torch.zeros_like(v) for k,v in full.items()}),('full',full)]:
        set_core_state(route,values);prior=rows(baseline/task/state/'predictions.jsonl')
        for i,record in enumerate(records[:16]):
            if task=='understanding':
                answer=adapter.understand(Path(record['user_image']),record['user_prompt'],max_new_tokens=32)
                passed=answer==prior[i]['prediction']
            else:
                image=fresh.generate(record['user_prompt'],seed=8087+i,**kwargs)
                dest=out/'replay'/task/state;dest.mkdir(parents=True,exist_ok=True)
                image.save(dest/f'{i:05d}.png')
                with Image.open(baseline/task/state/prior[i]['image']) as previous:passed=same_pixels(image,previous)
            replay.append(dict(state=state,index=i,split_key=record['split_key'],passed=passed))
            write(out/f'{task}-replay.json',replay)
            if not passed:raise ValueError(f'historical {task} {state} replay differs at {i}')
    set_core_state(route,negated_state(full));dest=out/task/'negated'
    write(out/'status.json',dict(status='evaluating',task=task,state='negated',updated=time.time()))
    if task=='understanding':
        generate_understanding_predictions(adapter,records,dest,understanding_kwargs={'max_new_tokens':32},
            metadata=dict(condition='negated',source_seed=seed),resume=True)
        validate_completed_predictions(records,rows(dest/'predictions.jsonl'))
    else:
        generate_fixed_predictions(fresh,records,dest,seed=8087,generation_kwargs=kwargs,
                                   metadata=dict(condition='negated',source_seed=seed),resume=True)


def run(root,capability,seed):
    from scripts.audit_reference_generation import audit_panel
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.source_measurement import audit_rows
    trajectory=root/'runs/mechanism_controls/fixed_core_trajectory_v1'
    qualification=read(trajectory/'qualification.json');matched=family_status(qualification,capability)
    if read(trajectory/capability/f'seed-{seed}/complete.json').get('complete') is not True:
        raise ValueError('trajectory job incomplete')
    core,baseline,u,g,gpath,evidence=baseline_evidence(root,capability,seed)
    out=root/f'runs/mechanism_controls/fixed_core_endpoints_v1/{capability}/seed-{seed}'
    lock_json(out/'design.json',dict(core=str(core),baseline=str(baseline),capability=capability,seed=seed,
        family_trajectory_matched=matched,qualification=qualification['capabilities'][capability],
        understanding_records=u,generation_records=g,control='every Q is replaced by -Q; no fitted scale',
        generation_kwargs=dict(height=512,width=512,diffusion_steps=30,guidance_scale=3.0),
        understanding_max_new_tokens=32,inference_seed=8087,cache_policy='reset per image',
        baseline_replay='first 16 input-ordered questions/images in each full and zero state; stop on any mismatch',
        baseline_reuse='entire historical panels, all input identities and raw official scores checked',
        inclusion='all eight registered endpoints, including any unmatched families and every effect sign'))
    for task,records in [('understanding',u),('generation',g)]:
        infer_task(root,out,core,baseline,capability,seed,task,records)
        import torch
        gc.collect();torch.cuda.empty_cache()
    target=out/'generation/negated';stage=target/'official-stage'
    write(out/'status.json',dict(status='official_scoring',updated=time.time()))
    py=root/'environments/t2i_compbench/.venv/bin/python'
    if not (stage/'staging_manifest.json').exists():
        subprocess.run([str(root/'environments/blip3o/.venv/bin/python'),'-P',str(root/'scripts/stage_oral_predictions.py'),
            '--records',str(gpath),'--predictions',str(target/'predictions.jsonl'),'--output',str(stage)],check=True)
    summary=stage/'t2i_compbench/official-summary.json'
    if not summary.exists():
        subprocess.run([str(py),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
            '--repository',str(root/'external/T2I-CompBench'),'--lock',str(root/'configs/evaluators.lock.json'),
            '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),'--python',str(py),'--no-revision-verification'],check=True)
    evidence['understanding']['negated']=audit_rows(rows(out/'understanding/negated/predictions.jsonl'))[0]
    evidence['generation']['negated']=audit_panel(target,g,'negated',512)
    keys=lambda p:[(v['split_key'],v['benchmark_id'],v['category']) for v in p['items']]
    if keys(evidence['generation']['zero'])!=keys(evidence['generation']['negated']) or keys(evidence['generation']['full'])!=keys(evidence['generation']['negated']):
        raise ValueError('scoring indices are not paired across states')
    uvalues={s:100*r['legacy_accuracy'] for s,r in evidence['understanding'].items()}
    gvalues={s:r['score_x100'] for s,r in evidence['generation'].items()}
    contrasts={f'{a}_minus_{b}':dict(understanding_pp=uvalues[a]-uvalues[b],generation_score_x100=gvalues[a]-gvalues[b])
               for a,b in [('full','negated'),('negated','zero'),('full','zero')]}
    write(out/'complete.json',dict(complete=True,capability=capability,seed=seed,family_trajectory_matched=matched,
        evidence=evidence,contrasts=contrasts,scope='fixed learned-core direction control; not semantic nuisance replacement'))
    write(out/'status.json',dict(status='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capability',choices=('counting','spatial'),required=True)
    p.add_argument('--seed',type=int,choices=(17,29,43,71),required=True)
    a=p.parse_args();root=a.root.resolve()
    from skillbridge.original_core_audit import admit_gpu
    from skillbridge.training_overlap import training_lock
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('registered GPU 3 required')
    out=root/f'runs/mechanism_controls/fixed_core_endpoints_v1/{a.capability}/seed-{a.seed}'
    with training_lock(root/'runs/mechanism_controls/gpu_serial_slots/gpu-3',blocking=False):
        try:
            for attempt in range(120):
                uid,free,util=[v.strip() for v in subprocess.check_output(['nvidia-smi','-i','3',
                    '--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
                processes=[]
                for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
                    gpu,pid,mem=[v.strip() for v in line.split(',')]
                    if gpu==uid:processes.append(dict(pid=int(pid),memory_mib=int(mem)))
                try:admit_gpu(3,int(free),int(util),processes,allowed_idle_pids=[]);break
                except ValueError:
                    if attempt==119:raise
                    time.sleep(5)
            run(root,a.capability,a.seed)
        except Exception as error:
            write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()));raise


if __name__=='__main__':main()
