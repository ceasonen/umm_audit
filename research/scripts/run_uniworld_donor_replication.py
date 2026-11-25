#!/usr/bin/env python3
"""Paired donor study at UniWorld's native Qwen-to-FLUX conditioning interface."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_uniworld_replication_endpoints import rows,write
from skillbridge.donor_replication import CONDITIONS,SEEDS,scoring_view,summarize_job,aggregate_jobs
from skillbridge.mechanism_sequence import lock_json


def registered_gpu(gpu,environ):
    if gpu not in (0,2,3) or environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
        raise ValueError('registered donor GPU must be authorized and match CUDA visibility')
    return gpu


def lines(path,values):
    content=''.join(json.dumps(value,sort_keys=True,allow_nan=False)+'\n' for value in values)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=content:raise ValueError('paired scoring records differ')
    if not path.exists():path.write_text(content)


def job_inputs(root,capability,seed):
    from scripts.run_blip3o_readout_followup import targets
    name='semantic_interchange_native_v2' if capability=='counting' else 'semantic_interchange_spatial_v2'
    source=root/f'runs/mechanism_controls/{name}/seed-17/design.json'
    pairs=json.loads(source.read_text())['pairs']
    if len(pairs)!=128:raise ValueError('fixed 128-pair donor panel required')
    for other_seed in SEEDS:
        another=root/f'runs/mechanism_controls/{name}/seed-{other_seed}/design.json'
        if json.loads(another.read_text())['pairs']!=pairs:raise ValueError('donor pairs differ across runs')
    base=root/f'runs/mechanism_controls/uniworld_core_replication_v1/{capability}/seed-{seed}'
    endpoints=base/'endpoints'
    completed=json.loads((endpoints/'complete.json').read_text())
    if completed.get('complete') is not True:raise ValueError('native endpoints incomplete')
    inputs=rows(targets(root,capability))
    if len(inputs)!=1000:raise ValueError('native generation panel differs')
    histories={};noises=None
    for state in ('zero','full'):
        bank={item['split_key']:item for item in rows(endpoints/'generation'/state/'predictions.jsonl')}
        histories[state]=[bank[pair['recipient']['split_key']] for pair in pairs]
        if any(item['prompt']!=pair['recipient']['user_prompt'] for item,pair in zip(histories[state],pairs)):
            raise ValueError('recipient prompt differs from native endpoint')
        seeds=[item['seed'] for item in histories[state]]
        if noises is None:noises=seeds
        elif noises!=seeds:raise ValueError('native baseline seeds differ')
    return dict(capability=capability,source_seed=seed,pairs=pairs,noises=noises,
                core=str(base/'core.pt'),reference=str(endpoints),histories=histories)


def generate(root,out,job,*,limit=None):
    import torch
    from PIL import Image
    from scripts.evaluate_cross_backbone_generation import load_uniworld_v1
    from skillbridge.adapter_checkpoint import load_capability_checkpoint
    from skillbridge.decoder_sanity import same_pixels
    from skillbridge.readout_probe import set_core_state
    from skillbridge.uniworld_donor import matched_change
    capability=job['capability'];seed=job['source_seed'];core=Path(job['core'])
    checkpoint=torch.load(core,map_location='cpu',weights_only=True)
    expected=dict(backbone='uniworld_v1',capability=capability,seed=seed,method='tied_core',rank=64)
    if any(checkpoint['metadata'].get(k)!=v for k,v in expected.items()):
        raise ValueError('UniWorld checkpoint identity differs')
    full=checkpoint['state_dict'];zero={k:torch.zeros_like(v) for k,v in full.items()}
    adapter,route,kwargs=load_uniworld_v1(root,rank=64,timestep_buckets=1,method='tied_core',
        gain_limit=.2 if capability=='counting' else .1,target_suffixes=('q_proj','v_proj'))
    kwargs['condition_path']='joint'
    if kwargs!=dict(height=512,width=512,diffusion_steps=28,guidance_scale=3.5,condition_path='joint'):
        raise ValueError('UniWorld native generation settings differ')
    load_capability_checkpoint(route,core)
    lock_json(out/'design.json',dict(job=job,conditions=list(CONDITIONS),generation_kwargs=kwargs,
        interface='Qwen denoise embeddings entering frozen FLUX; recipient T5 and pooled embeddings stay fixed',
        normalization='Frobenius norm after monotone tokenizer-position alignment of native Qwen change',sign_seed=1017,
        selection='all 128 registered pairs; retain every norm mismatch'))
    completed=[];pairs=job['pairs'][:limit] if limit is not None else job['pairs']
    for i,pair in enumerate(pairs):
        noise=job['noises'][i];recipient=pair['recipient']['user_prompt']
        unit=out/'units'/f'{i:05d}';identity=dict(index=i,pair=pair,noise=noise,core=str(core))
        saved=unit/'complete.json'
        if saved.exists():
            prior=json.loads(saved.read_text())
            if prior.get('complete') is not True or prior['identity']!=identity:
                raise ValueError('resume donor unit differs')
            completed.append(prior);continue
        unit.mkdir(parents=True,exist_ok=True)
        def capture(prompt,state):
            set_core_state(route,full if state=='full' else zero)
            carrier,ids=adapter.generation_condition(prompt,return_input_ids=True)
            return carrier.detach().clone(),ids
        base,recipient_ids=capture(recipient,'zero');updated,full_ids=capture(recipient,'full')
        if recipient_ids!=full_ids:raise ValueError('recipient tokenizer positions changed across core states')
        reference=updated.float()-base.float()
        controls={};audits={};norm_flags={}
        for kind in ('semantic_donor','nuisance_donor'):
            donor_prompt=pair[kind]['user_prompt']
            donor_zero,donor_ids=capture(donor_prompt,'zero');donor_full,donor_full_ids=capture(donor_prompt,'full')
            if donor_ids!=donor_full_ids:raise ValueError('donor tokenizer positions changed across core states')
            changed,audit=matched_change(base,updated,donor_zero,donor_full,
                                         recipient_token_ids=recipient_ids,donor_token_ids=donor_ids)
            controls[kind]=changed;audits[kind]=audit;norm_flags[kind]=audit['norm_matched']
        rng=torch.Generator(device=base.device).manual_seed(1017)
        signs=torch.randint(0,2,reference.shape,device=base.device,generator=rng).mul(2).sub(1)
        controls['random_sign']=(base.float()+reference*signs).to(base.dtype)
        reference_norm=float(reference.norm());sign_norm=float((controls['random_sign'].float()-base.float()).norm())
        norm_flags['random_sign']=abs(sign_norm/reference_norm-1)<=.2 if reference_norm else sign_norm==0
        audits['random_sign']=dict(reference_norm=reference_norm,actual_norm=sign_norm,norm_matched=norm_flags['random_sign'])
        controls.update(interface_replay=updated,conditional_only=updated,unconditional_only=base)
        predictions={}
        for condition in CONDITIONS:
            write(out/'status.json',dict(status='generating',index=i,condition=condition,updated=time.time()))
            destination=out/condition/'images'/f'{i:05d}.png';destination.parent.mkdir(parents=True,exist_ok=True)
            if condition in ('zero','full','conditional_only','unconditional_only','interface_replay'):
                source_state='full' if condition in ('full','conditional_only','interface_replay') else 'zero'
                source=Path(job['reference'])/'generation'/source_state/job['histories'][source_state][i]['image']
                if not destination.exists():os.link(source,destination)
                if i==0 and condition in ('zero','full','conditional_only','unconditional_only','interface_replay'):
                    set_core_state(route,zero)
                    if condition=='zero':
                        candidate=adapter.generate(recipient,seed=noise,**kwargs)
                    else:
                        fixed=controls[condition] if condition in controls else updated
                        candidate=adapter.generate(recipient,seed=noise,
                            condition_transform=lambda _value,fixed=fixed:fixed,**kwargs)
                    with Image.open(source) as historical:
                        if not same_pixels(candidate,historical):raise ValueError('native baseline/interface replay differs')
            else:
                set_core_state(route,zero);fixed=controls[condition]
                candidate=adapter.generate(recipient,seed=noise,
                    condition_transform=lambda _value,fixed=fixed:fixed,**kwargs)
                candidate.save(destination)
            predictions[condition]=dict(split_key=pair['recipient']['split_key'],prompt=recipient,
                seed=noise,source_seed=seed,condition=condition,image=f'images/{i:05d}.png')
        report=dict(complete=True,identity=identity,predictions=predictions,
                    norm_flags=norm_flags,audits=audits)
        write(saved,report);completed.append(report)
    if limit is None:
        for condition in CONDITIONS:
            lines(out/condition/'predictions.jsonl',[row['predictions'][condition] for row in completed])
        write(out/'generation-complete.json',dict(complete=True,rows=128,conditions=list(CONDITIONS),
            norm_flags={kind:[row['norm_flags'][kind] for row in completed]
                        for kind in ('semantic_donor','nuisance_donor','random_sign')}))
    else:
        write(out/'smoke-complete.json',dict(complete=True,rows=len(completed),norm_flags=completed[0]['norm_flags']))


def score(root,out,job):
    from scripts.audit_reference_generation import audit_panel
    generation=json.loads((out/'generation-complete.json').read_text())
    if generation.get('complete') is not True or generation['rows']!=128:
        raise ValueError('donor generation incomplete')
    panels={};scorer=root/'environments/t2i_compbench/.venv/bin/python'
    for condition in CONDITIONS:
        predictions=rows(out/condition/'predictions.jsonl');panels[condition]={}
        for criterion in ('recipient','donor'):
            write(out/'status.json',dict(status='scoring',condition=condition,criterion=criterion,updated=time.time()))
            records,outputs=scoring_view(job['pairs'],predictions,out/condition,
                criterion=criterion,condition=condition,source_seed=job['source_seed'],
                noises=job['noises'],capability=job['capability'])
            target=out/'scored'/condition/criterion
            lines(target/'records.jsonl',records);lines(target/'predictions.jsonl',outputs)
            stage=target/'official-stage'
            if not (stage/'staging_manifest.json').exists():
                subprocess.run([str(root/'environments/uniworld_v1/.venv/bin/python'),'-P',
                    str(root/'scripts/stage_oral_predictions.py'),'--records',str(target/'records.jsonl'),
                    '--predictions',str(target/'predictions.jsonl'),'--output',str(stage)],check=True)
            summary=stage/'t2i_compbench/official-summary.json'
            if not summary.exists():
                subprocess.run([str(scorer),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                    '--repository',str(root/'external/T2I-CompBench'),
                    '--lock',str(root/'configs/evaluators.lock.json'),
                    '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),
                    '--python',str(scorer),'--no-revision-verification'],check=True)
            panels[condition][criterion]=audit_panel(target,records,condition,512,
                                                      noise_seeds=job['noises'])
    report=summarize_job(job['capability'],job['source_seed'],panels,generation['norm_flags'])
    report.update(input_pairs=job['pairs'],panels=panels,
                  scope='Qwen denoise interface; paired recipient T5 and frozen FLUX')
    write(out/'complete.json',report)
    write(out/'status.json',dict(status='complete',updated=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--capability',choices=('counting','spatial'),required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-units',type=int,default=1)
    parser.add_argument('--gpu',type=int,choices=(0,2,3),default=0)
    args=parser.parse_args();root=args.root.resolve()
    gpu=registered_gpu(args.gpu,os.environ)
    from scripts.run_extension_campaign import idle
    from skillbridge.training_overlap import training_lock
    out=root/f'runs/mechanism_controls/uniworld_donor_replication_v1/{args.capability}/seed-{args.seed}'
    with training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{gpu}',blocking=False):
        if not idle(gpu,[gpu]):raise ValueError(f'GPU{gpu} is not idle')
        try:
            job=job_inputs(root,args.capability,args.seed)
            if args.smoke and not 1<=args.smoke_units<=128:raise ValueError('smoke units outside fixed panel')
            generate(root,out,job,limit=args.smoke_units if args.smoke else None)
            if not args.smoke:
                import torch
                gc.collect();torch.cuda.empty_cache()
                score(root,out,job)
        except Exception as error:
            write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()))
            raise


if __name__=='__main__':main()
