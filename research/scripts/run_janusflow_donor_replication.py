#!/usr/bin/env python3
"""Complete fixed-panel JanusFlow donor generation and paired attribute scoring."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time

from scripts.run_fixed_core_endpoints import read,rows,write
from skillbridge.donor_replication import CONDITIONS,SEEDS,scoring_view,summarize_job,aggregate_jobs
from skillbridge.mechanism_sequence import lock_json


def write_lines(path,values):
    import json
    text=''.join(json.dumps(v,sort_keys=True,allow_nan=False)+'\n' for v in values)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:raise ValueError('recorded scoring inputs differ')
    if not path.exists():path.write_text(text)


def job_inputs(root,capability,seed):
    from scripts.audit_reference_generation import audit_panel
    name='semantic_interchange_native_v2' if capability=='counting' else 'semantic_interchange_spatial_v2'
    pair_source=root/f'runs/mechanism_controls/{name}/seed-17/design.json'
    pairs=read(pair_source)['pairs']
    if len(pairs)!=128:raise ValueError('fixed 128-pair panel required')
    # All four completed BLIP3-o panels must use these same input pairs.
    for source_seed in SEEDS:
        if read(root/f'runs/mechanism_controls/{name}/seed-{source_seed}/design.json')['pairs']!=pairs:
            raise ValueError('BLIP3-o donor inputs differ across runs')
    reference=root/f'runs/mechanism_controls/mechanism_sequence_v1/janusflow/{capability}/seed-{seed}/replication'
    core=reference/'core.pt'
    if not core.is_file():raise ValueError('reference checkpoint missing')
    filename='records.jsonl' if capability=='counting' else 'records-balanced-1000.jsonl'
    records=rows(root/f'data/oral_benchmarks_confirmatory_v1/{capability}/tiers/confirmatory/{filename}')
    design=read(reference/'endpoints/base/generation-design.json')
    expected=dict(core=str(core),checkpoint_method='tied_core',checkpoint_capability=capability,
        test_capability=capability,backbone='janusflow',source_seed=seed,inference_seed=8087,
        generation_kwargs=dict(image_size=384,steps=30,cfg_scale=5.0),cache_policy='reset per image')
    if any(design.get(k)!=v for k,v in expected.items()):raise ValueError('native baseline protocol differs')
    if len(records)!=1000:raise ValueError('baseline input panel differs')
    histories={};evidence={}
    for state in ('zero','full'):
        folder=reference/f'endpoints/base/generation/{state}'
        evidence[state]=audit_panel(folder,records,state,384)
        indexed={p['split_key']:p for p in rows(folder/'predictions.jsonl')}
        histories[state]=[]
        for pair in pairs:
            p=indexed[pair['recipient']['split_key']]
            if p['prompt']!=pair['recipient']['user_prompt']:raise ValueError('recipient prompt differs')
            histories[state].append(p)
    noises=[p['seed'] for p in histories['zero']]
    if noises!=[p['seed'] for p in histories['full']]:raise ValueError('baseline noise pairing differs')
    return dict(capability=capability,source_seed=seed,pairs=pairs,noises=noises,
                core=str(core),reference=str(reference),histories=histories,baseline_evidence=evidence)


def generate(root,out,job):
    import torch
    from PIL import Image
    from scripts.evaluate_cross_backbone_generation import load_janusflow,checkpoint_route_spec
    from skillbridge.janusflow_donor import SameLatentDonor,model_capture
    from skillbridge.readout_probe import set_core_state
    from skillbridge.decoder_sanity import same_pixels
    cap=job['capability'];seed=job['source_seed'];core=Path(job['core'])
    lock_json(out/'design.json',dict(job=job,conditions=list(CONDITIONS),sign_seed=1017,
        baseline_reuse='all input identities and official scores checked; first 16 native zero/full images replayed',
        interface_replay='all 128 full-interface outputs must match historical full-core pixels',
        selection='all fixed 128 input pairs; retain every control and norm mismatch',
        donor='same current recipient latent/time; frozen recipient unconditional branch',
        normalization='per-step conditional update norm; audit actual update after casting'))
    checkpoint=torch.load(core,map_location='cpu',weights_only=True)
    expected=dict(backbone='janusflow',capability=cap,seed=seed,rank=64,method='tied_core')
    if any(checkpoint['metadata'].get(k)!=v for k,v in expected.items()):raise ValueError('checkpoint metadata differs')
    full=checkpoint['state_dict']
    if len(full)!=48 or any(tuple(v.shape)!=(64,64) for v in full.values()):raise ValueError('48 rank-64 cores required')
    gain,suffixes=checkpoint_route_spec(core)
    torch.cuda.set_per_process_memory_fraction(.85,0);torch.manual_seed(8087)
    adapter,route,kwargs=load_janusflow(root,rank=64,timestep_buckets=1,method='tied_core',gain_limit=gain,target_suffixes=suffixes)
    if kwargs!=dict(image_size=384,steps=30,cfg_scale=5.0):raise ValueError('generation settings differ')
    adapter.model.eval();zero={k:torch.zeros_like(v) for k,v in full.items()}
    capture=model_capture(adapter,route,full);completed=[]
    for i,(pair,noise) in enumerate(zip(job['pairs'],job['noises'],strict=True)):
        unit=out/'units'/f'{i:05d}';unit.mkdir(parents=True,exist_ok=True)
        identity=dict(index=i,pair=pair,noise=noise,core=str(core))
        if (unit/'complete.json').exists():
            prior=read(unit/'complete.json')
            if prior['identity']!=identity or prior.get('complete') is not True:raise ValueError('resume unit differs')
            for c in CONDITIONS:
                with Image.open(out/c/prior['predictions'][c]['image']) as im:
                    if im.size!=(384,384):raise ValueError('resume image differs')
            completed.append(prior);continue
        predictions={};norm_flags={};prompt=pair['recipient']['user_prompt']
        for c in CONDITIONS:
            write(out/'status.json',dict(status='generating',index=i,condition=c,updated=time.time()))
            dest=out/c/'images'/f'{i:05d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
            set_core_state(route,full if c=='full' else zero)
            if c in ('zero','full'):
                old=Path(job['reference'])/f'endpoints/base/generation/{c}'/job['histories'][c][i]['image']
                with Image.open(old) as previous:
                    if i<16:
                        replay=adapter.generate(prompt,seed=noise,**kwargs)
                        replay.save(unit/f'{c}-native-replay.png')
                        if not same_pixels(replay,previous):raise ValueError('live native baseline replay differs')
                if dest.exists():
                    if not dest.samefile(old) and dest.read_bytes()!=old.read_bytes():raise ValueError('baseline reuse differs')
                else:os.link(old,dest)
            else:
                role='nuisance_donor' if c=='nuisance_donor' else 'semantic_donor'
                intervention=SameLatentDonor(capture,prompt,pair[role]['user_prompt'],c,sign_seed=1017)
                image=adapter.generate(prompt,seed=noise,step_carrier_transform=intervention,**kwargs)
                image.save(dest)
                if len(intervention.audit)!=30:raise ValueError('complete 30-step audit required')
                if c in ('semantic_donor','nuisance_donor','random_sign','conditional_only') and not all(r['unconditional_unchanged'] for r in intervention.audit):
                    raise ValueError('unconditional carrier changed')
                if c=='interface_replay':
                    with Image.open(out/'full/images'/f'{i:05d}.png') as previous:
                        if not same_pixels(image,previous):raise ValueError('full-interface replay differs')
                write(unit/f'{c}-audit.json',intervention.audit)
                if c in ('semantic_donor','nuisance_donor','random_sign'):
                    norm_flags[c]=all(r['norm_matched'] for r in intervention.audit)
            predictions[c]=dict(split_key=pair['recipient']['split_key'],prompt=prompt,seed=noise,
                source_seed=seed,condition=c,image=f'images/{i:05d}.png')
        report=dict(complete=True,identity=identity,predictions=predictions,norm_flags=norm_flags,
                    native_replays=2 if i<16 else 0,full_interface_replay=True)
        write(unit/'complete.json',report);completed.append(report)
    for c in CONDITIONS:write_lines(out/c/'predictions.jsonl',[r['predictions'][c] for r in completed])
    write(out/'generation-complete.json',dict(complete=True,rows=128,conditions=list(CONDITIONS),
        native_replays=32,full_interface_replays=128,
        norm_flags={c:[r['norm_flags'][c] for r in completed] for c in ('semantic_donor','nuisance_donor','random_sign')}))


def score(root,out,job):
    from scripts.audit_reference_generation import audit_panel
    generation=read(out/'generation-complete.json')
    if generation.get('complete') is not True or generation['rows']!=128:raise ValueError('generation incomplete')
    panels={}
    py=root/'environments/t2i_compbench/.venv/bin/python'
    for c in CONDITIONS:
        predictions=rows(out/c/'predictions.jsonl');panels[c]={}
        for criterion in ('recipient','donor'):
            write(out/'status.json',dict(status='scoring',condition=c,criterion=criterion,updated=time.time()))
            records,pred=scoring_view(job['pairs'],predictions,out/c,criterion=criterion,condition=c,
                source_seed=job['source_seed'],noises=job['noises'],capability=job['capability'])
            target=out/'scored'/c/criterion
            write_lines(target/'records.jsonl',records);write_lines(target/'predictions.jsonl',pred)
            stage=target/'official-stage'
            if not (stage/'staging_manifest.json').exists():
                subprocess.run([str(root/'environments/janusflow/.venv/bin/python'),'-P',str(root/'scripts/stage_oral_predictions.py'),
                    '--records',str(target/'records.jsonl'),'--predictions',str(target/'predictions.jsonl'),'--output',str(stage)],check=True)
            summary=stage/'t2i_compbench/official-summary.json'
            if not summary.exists():
                subprocess.run([str(py),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                    '--repository',str(root/'external/T2I-CompBench'),'--lock',str(root/'configs/evaluators.lock.json'),
                    '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),'--python',str(py),'--no-revision-verification'],check=True)
            panels[c][criterion]=audit_panel(target,records,c,384,noise_seeds=job['noises'])
    report=summarize_job(job['capability'],job['source_seed'],panels,generation['norm_flags'])
    report.update(input_pairs=job['pairs'],panels=panels,scope='dynamic donor conditioning at recipient latent; all fixed inputs retained')
    write(out/'complete.json',report);write(out/'status.json',dict(status='complete',updated=time.time()))


def preflight(root):
    out=root/'runs/mechanism_controls/janusflow_donor_replication_v1'
    jobs=[job_inputs(root,c,s) for s in SEEDS for c in ('counting','spatial')]
    lock_json(out/'plan.json',dict(jobs=[dict(capability=j['capability'],source_seed=j['source_seed'],
        core=j['core'],pairs=j['pairs'],noises=j['noises']) for j in jobs],conditions=list(CONDITIONS),
        total_scored_images=8192,new_condition_images=6144,reused_baseline_images=2048,
        additional_native_replay_images=256,attribute_scores=16384,
        control_inclusion='all eight conditions and every unit, including norm failures',
        interface='dynamic visual carrier; donor evaluated at the current recipient latent and time',
        independence='four training runs; repeated questions and shared donors are dependent',gpu=1))
    write(out/'preflight.json',dict(complete=True,jobs=8,baseline_scores_checked=16000,
        fixed_pairs_per_capability=128,checked_at=time.time()))


def queue(root,pid):
    from scripts.queue_matched_backbone_source_audit import worker_identity
    from skillbridge.training_overlap import training_lock
    out=root/'runs/mechanism_controls/janusflow_donor_replication_v1'
    with training_lock(out/'queue-lock',blocking=False):
        def status(s,**kw):write(out/'queue.json',dict(status=s,updated=time.time(),**kw))
        try:
            if read(out/'preflight.json').get('complete') is not True:raise ValueError('full preflight required')
            prior=worker_identity(pid)
            if prior and not any(v.endswith('/run_janusflow_donor_validation.py') for v in prior['argv']):raise ValueError('validation predecessor differs')
            status('waiting_for_validation',pid=pid,predecessor=prior);deadline=time.monotonic()+24*3600
            while prior is not None:
                current=worker_identity(pid)
                if current is None or current['process_start_ticks']!=prior['process_start_ticks']:break
                if current!=prior:raise ValueError('validation process identity changed')
                if time.monotonic()>deadline:status('expired');return
                time.sleep(30)
            validation=root/'runs/mechanism_controls/janusflow_donor_validation_v1'
            result=read(validation/'complete.json')
            if read(validation/'queue.json').get('status')!='complete' or any(result.get(k)!=v for k,v in dict(complete=True,units=4,images=32,historical_replays=8,full_interface_replays=4).items()):
                raise ValueError('real-model validation incomplete')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=f'{root}/src:{root}',HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',DIFFUSERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            reports=[]
            for seed in SEEDS:
                for capability in ('counting','spatial'):
                    cmd=[str(root/'environments/janusflow/.venv/bin/python'),'-u','-P',str(root/'scripts/run_janusflow_donor_replication.py'),
                        '--root',str(root),'--capability',capability,'--seed',str(seed)]
                    with (out/f'{capability}-{seed}.log').open('a') as log:
                        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                        status('running',capability=capability,seed=seed,child_pid=child.pid)
                        code=child.wait()
                    if code:raise RuntimeError(f'{capability}/{seed} failed: {code}')
                    reports.append(read(out/capability/f'seed-{seed}/complete.json'))
            write(out/'complete.json',dict(complete=True,reports=reports,aggregates=aggregate_jobs(reports)))
            status('complete')
        except Exception as error:status('failed',error=str(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--validation-queue-pid',type=int)
    p.add_argument('--capability',choices=('counting','spatial'));p.add_argument('--seed',type=int,choices=SEEDS)
    a=p.parse_args();root=a.root.resolve()
    if a.preflight:preflight(root);return
    if a.validation_queue_pid is not None:queue(root,a.validation_queue_pid);return
    if a.capability is None or a.seed is None:p.error('capability and seed required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('registered GPU1 required')
    from skillbridge.training_overlap import training_lock
    from skillbridge.original_core_audit import admit_gpu
    out=root/f'runs/mechanism_controls/janusflow_donor_replication_v1/{a.capability}/seed-{a.seed}'
    with training_lock(root/'runs/mechanism_controls/gpu_serial_slots/gpu-1',blocking=False):
        try:
            for attempt in range(120):
                uid,free,util=[v.strip() for v in subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
                processes=[]
                for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
                    gpu,pid,mem=[v.strip() for v in line.split(',')]
                    if gpu==uid:processes.append(dict(pid=int(pid),memory_mib=int(mem)))
                try:admit_gpu(1,int(free),int(util),processes,allowed_idle_pids=[]);break
                except ValueError:
                    if attempt==119:raise
                    time.sleep(5)
            job=job_inputs(root,a.capability,a.seed)
            generate(root,out,job)
            import torch
            gc.collect();torch.cuda.empty_cache()
            score(root,out,job)
        except Exception as error:write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()));raise


if __name__=='__main__':main()
