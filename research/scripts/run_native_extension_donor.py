#!/usr/bin/env python3
"""Native paired donor interfaces for Janus-Pro and Show-o2 reference cores."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_uniworld_replication_endpoints import rows,write
from skillbridge.donor_replication import CONDITIONS,SEEDS,scoring_view,summarize_job
from skillbridge.mechanism_sequence import lock_json
from skillbridge.native_extension_endpoints import endpoint_plan,validate_training_configuration


def registered_donor_gpu(model, gpu, environ):
    allowed = {'januspro': (0, 2, 4, 5, 7), 'showo2': (2, 5), 'dimoo': (1, 3)}
    if gpu not in allowed.get(model, ()) or environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('unregistered native donor GPU')
    return gpu


def selected_pairs(pairs, *, start=0, end=None, limit=None):
    if not isinstance(start, int) or not 0 <= start < len(pairs) or \
            (end is not None and (not isinstance(end, int) or not start < end <= len(pairs))) or \
            (limit is not None and (start != 0 or end is not None or limit != 1)):
        raise ValueError('invalid native donor pair shard')
    return list(enumerate(pairs[start:end][:limit] if limit is not None else pairs[start:end], start))


def native_plan(model):
    if model=='dimoo':
        from skillbridge.dimoo_reference_evaluation import campaign_plan
        return dict(image_size=512,generation_kwargs=campaign_plan()['generation_kwargs'])
    return endpoint_plan(model)


def lines(path,values):
    content=''.join(json.dumps(value,sort_keys=True,allow_nan=False)+'\n' for value in values)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=content:raise ValueError('donor scoring records differ')
    if not path.exists():path.write_text(content)


def job_inputs(root,model,capability,seed):
    name='semantic_interchange_native_v2' if capability=='counting' else 'semantic_interchange_spatial_v2'
    pairs=json.loads((root/f'runs/mechanism_controls/{name}/seed-17/design.json').read_text())['pairs']
    if len(pairs)!=128:raise ValueError('fixed 128-pair donor panel required')
    for s in SEEDS:
        source=json.loads((root/f'runs/mechanism_controls/{name}/seed-{s}/design.json').read_text())['pairs']
        if source!=pairs:raise ValueError('donor pair selection differs across runs')
    base=root/f'runs/mechanism_controls/{model}_reference_replication_v1/{capability}/seed-{seed}'
    if model=='dimoo':
        from scripts.run_dimoo_reference_replication import validate_training
        campaign=json.loads((root/'runs/mechanism_controls/dimoo_reference_replication_v1/design.json').read_text())
        training_job=next(item for item in campaign['jobs']
                          if item['capability']==capability and item['seed']==seed)
        if not validate_training(training_job):raise ValueError('DiMOO native training incomplete')
    else:
        validate_training_configuration(json.loads((base/'training-complete.json').read_text())['configuration'],
                                        model,capability,seed)
    endpoint=base/'endpoints'
    completed=json.loads((endpoint/'complete.json').read_text())
    expected_complete=dict(complete=True,capability=capability,seed=seed)
    if model!='dimoo':expected_complete['backbone']=model
    if any(completed.get(k)!=v for k,v in expected_complete.items()):
        raise ValueError('native endpoint is not complete for this checkpoint')
    parity=json.loads((endpoint/'zero-bypass-parity.json').read_text())
    answer_key='answer_tokens_equal' if model=='dimoo' else 'answer_equal'
    if parity.get(answer_key) is not True or parity.get('pixels_equal') is not True:
        raise ValueError('native endpoint zero-bypass parity missing')
    histories={};noises=None
    for state in ('zero','full'):
        bank={item['split_key']:item for item in rows(endpoint/'generation'/state/'predictions.jsonl')}
        histories[state]=[bank[pair['recipient']['split_key']] for pair in pairs]
        if any(item['prompt']!=pair['recipient']['user_prompt'] for item,pair in zip(histories[state],pairs)):
            raise ValueError('native donor recipient prompt differs')
        observed=[item['seed'] for item in histories[state]]
        if noises is None:noises=observed
        elif noises!=observed:raise ValueError('native recipient generation seeds differ')
    return dict(model=model,capability=capability,source_seed=seed,pairs=pairs,
                noises=noises,core=str(base/'core.pt'),reference=str(endpoint),histories=histories)


def load_native(root,job,*,postcast_calibration=False):
    import torch
    from scripts.train_cross_backbone_pilot import attach_route
    from skillbridge.adapter_checkpoint import load_capability_checkpoint
    model_name=job['model'];capability=job['capability'];core=Path(job['core'])
    checkpoint=torch.load(core,map_location='cpu',weights_only=True)
    expected=dict(backbone=model_name,capability=capability,seed=job['source_seed'],
                  method='tied_core',rank=64,timestep_buckets=1)
    full=checkpoint['state_dict']
    if any(checkpoint['metadata'].get(k)!=v for k,v in expected.items()) or not full or \
            any(tuple(v.shape)!=(64,64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('native donor checkpoint identity or tensor shape differs')
    if model_name=='dimoo':
        from scripts.evaluate_cross_backbone_generation import load_dimoo
        from skillbridge.dimoo_native_donor import DimooDonorGenerator
        adapter,route,kwargs=load_dimoo(root,rank=64,timestep_buckets=1,method='tied_core',
            gain_limit=.2 if capability=='counting' else .1,target_suffixes=('q_proj','v_proj'))
        if kwargs!=native_plan(model_name)['generation_kwargs']:
            raise ValueError('DiMOO native decoder differs')
        load_capability_checkpoint(route,core)
        device=next(route.parameters()).device
        full={key:value.to(device=device) for key,value in full.items()}
        return adapter,DimooDonorGenerator(adapter,full,postcast_calibration),route,full
    if model_name=='januspro':
        from skillbridge.januspro_runtime import load_components,JanusProInferenceAdapter
        from skillbridge.januspro_native_donor import JanusProDonorGenerator
        model,processor=load_components(root);route=model.language_model.model
        components=(model,processor)
    else:
        from skillbridge.showo2_runtime import load_components,Showo2InferenceAdapter
        from skillbridge.showo2_native_donor import Showo2DonorGenerator
        model,tokenizer,ids,vae=load_components(root);route=model.showo.model
        components=(model,tokenizer,ids,vae)
    attach_route(route,rank=64,timestep_buckets=1,method='tied_core',
                 gain_limit=.2 if capability=='counting' else .1,target_suffixes=('q_proj','v_proj'))
    load_capability_checkpoint(route,core)
    device=next(route.parameters()).device
    full={key:value.to(device=device) for key,value in full.items()}
    adapter=(JanusProInferenceAdapter(*components,route) if model_name=='januspro'
             else Showo2InferenceAdapter(*components,route))
    donor=(JanusProDonorGenerator(adapter,full) if model_name=='januspro'
           else Showo2DonorGenerator(adapter,full))
    return adapter,donor,route,full


def generate(root,out,job,*,limit=None,start=0,end=None,postcast_calibration=False):
    import torch
    from PIL import Image
    from skillbridge.decoder_sanity import same_pixels
    from skillbridge.readout_probe import set_core_state
    model=job['model'];plan=native_plan(model)
    adapter,donor,route,full=load_native(root,job,
                                         postcast_calibration=postcast_calibration)
    zero={key:torch.zeros_like(value) for key,value in full.items()}
    design=dict(job=job,conditions=list(CONDITIONS),
        generation_kwargs=plan['generation_kwargs'],sign_seed=1017,
        interface='same recipient image-token history' if model=='januspro' else 'same recipient latent and flow time',
        normalization='conditional prediction change norm; unconditional branch retained',
        selection='all fixed 128 pairs, all norm mismatches retained')
    if postcast_calibration:design['postcast_calibration']=True
    lock_json(out/'design.json',design)
    selected=selected_pairs(job['pairs'],start=start,end=end,limit=limit)
    completed=[]
    for i,pair in selected:
        noise=job['noises'][i];prompt=pair['recipient']['user_prompt']
        unit=out/'units'/f'{i:05d}';identity=dict(index=i,pair=pair,noise=noise,core=job['core'])
        saved=unit/'complete.json'
        if saved.exists():
            prior=json.loads(saved.read_text())
            if prior.get('complete') is not True or prior['identity']!=identity:
                raise ValueError('native donor resume unit differs')
            completed.append(prior);continue
        unit.mkdir(parents=True,exist_ok=True)
        outputs={};norm_flags={};audit_paths={}
        for condition in CONDITIONS:
            write(out/'status.json',dict(status='generating',index=i,condition=condition,updated=time.time()))
            destination=out/condition/'images'/f'{i:05d}.png';destination.parent.mkdir(parents=True,exist_ok=True)
            if condition in ('zero','full'):
                source=Path(job['reference'])/'generation'/condition/job['histories'][condition][i]['image']
                if not destination.exists():os.link(source,destination)
                if i==0:
                    set_core_state(route,zero if condition=='zero' else full)
                    live=adapter.generate(prompt,seed=noise,**plan['generation_kwargs'])
                    with Image.open(source) as historical:
                        if not same_pixels(live,historical):raise ValueError('native baseline image replay differs')
            else:
                source_name=('nuisance_donor' if condition=='nuisance_donor' else 'semantic_donor')
                donor_prompt=pair[source_name]['user_prompt']
                native_kwargs=(dict(cfg_weight=5.,temperature=1.,image_token_num_per_image=576)
                               if model=='januspro' else
                               dict(guidance_scale=7.5,num_inference_steps=50) if model=='showo2'
                               else dict(height=512,width=512,steps=64,cfg_scale=4.,temperature=1.))
                image,audit=donor.generate(prompt,donor_prompt,condition,seed=noise,sign_seed=1017,**native_kwargs)
                image.save(destination)
                expected_steps=576 if model=='januspro' else 49 if model=='showo2' else 64
                if len(audit)!=expected_steps:raise ValueError('native donor step audit incomplete')
                if condition in ('semantic_donor','nuisance_donor','random_sign'):
                    norm_flags[condition]=all(row['norm_matched'] for row in audit)
                    if not all(row['unconditional_unchanged'] for row in audit):
                        raise ValueError('native donor unconditional branch changed')
                if condition=='interface_replay':
                    historical=Path(job['reference'])/'generation/full'/job['histories']['full'][i]['image']
                    with Image.open(historical) as baseline:
                        if not same_pixels(image,baseline):raise ValueError('full-interface replay differs')
                audit_file=unit/f'{condition}-audit.json';write(audit_file,audit)
                audit_paths[condition]=str(audit_file.relative_to(out))
            outputs[condition]=dict(split_key=pair['recipient']['split_key'],prompt=prompt,
                seed=noise,source_seed=job['source_seed'],condition=condition,image=f'images/{i:05d}.png')
        result=dict(complete=True,identity=identity,predictions=outputs,
                    norm_flags=norm_flags,audit_paths=audit_paths)
        write(saved,result);completed.append(result)
    del adapter,donor,route;gc.collect();torch.cuda.empty_cache()
    if start == 0 and end is None and limit is None:
        for condition in CONDITIONS:
            lines(out/condition/'predictions.jsonl',[r['predictions'][condition] for r in completed])
        write(out/'generation-complete.json',dict(complete=True,rows=128,conditions=list(CONDITIONS),
            norm_flags={condition:[r['norm_flags'][condition] for r in completed]
                        for condition in ('semantic_donor','nuisance_donor','random_sign')}))
    elif limit is not None:
        write(out/'smoke-complete.json',dict(complete=True,rows=len(completed),
                                             norm_flags=completed[0]['norm_flags']))
    else:
        write(out/'shards'/f'{start}-{end}-complete.json',dict(complete=True,
            start=start,end=end,rows=len(completed),source_seed=job['source_seed'],
            model=model,capability=job['capability']))


def score(root,out,job):
    from scripts.audit_reference_generation import audit_panel
    generation=json.loads((out/'generation-complete.json').read_text())
    if generation.get('complete') is not True or generation['rows']!=128:
        raise ValueError('native donor generation incomplete')
    size=native_plan(job['model'])['image_size']
    scorer=root/'environments/t2i_compbench/.venv/bin/python'
    panels={}
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
                subprocess.run([str(root/f'environments/{job["model"]}/.venv/bin/python'),'-P',
                    str(root/'scripts/stage_oral_predictions.py'),'--records',str(target/'records.jsonl'),
                    '--predictions',str(target/'predictions.jsonl'),'--output',str(stage)],check=True)
            summary=stage/'t2i_compbench/official-summary.json'
            if not summary.exists():
                subprocess.run([str(scorer),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                    '--repository',str(root/'external/T2I-CompBench'),
                    '--lock',str(root/'configs/evaluators.lock.json'),
                    '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),
                    '--python',str(scorer),'--no-revision-verification'],check=True)
            panels[condition][criterion]=audit_panel(target,records,condition,size,noise_seeds=job['noises'])
    report=summarize_job(job['capability'],job['source_seed'],panels,generation['norm_flags'])
    report.update(input_pairs=job['pairs'],panels=panels,
                  scope='native paired prediction carrier at fixed recipient generation state')
    write(out/'complete.json',report);write(out/'status.json',dict(status='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',choices=('januspro','showo2','dimoo'),required=True)
    p.add_argument('--capability',choices=('counting','spatial'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--gpu',type=int)
    p.add_argument('--shard-start',type=int)
    p.add_argument('--shard-end',type=int)
    p.add_argument('--postcast-calibration',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    primary_gpu={'januspro':4,'showo2':5,'dimoo':1}[a.model]
    gpu=primary_gpu if a.gpu is None else a.gpu
    registered_donor_gpu(a.model,gpu,os.environ)
    shard=a.shard_start is not None or a.shard_end is not None
    permitted_shards={('januspro','counting',17,2,64,128),
                      ('januspro','spatial',17,5,64,128),
                      ('dimoo','spatial',17,3,64,128)}
    if shard and (a.model,a.capability,a.seed,gpu,a.shard_start,a.shard_end) not in permitted_shards:
        raise ValueError('only registered deadline upper-half shards are allowed')
    if shard and a.smoke:raise ValueError('shard and smoke modes are exclusive')
    if a.postcast_calibration and (a.model!='dimoo' or a.seed!=17 or shard):
        raise ValueError('post-cast calibration is registered only for DiMOO seed 17')
    from scripts.run_extension_campaign import idle
    from scripts.run_dimoo_reference_replication import idle_gpu
    from skillbridge.training_overlap import training_lock
    family=(f'{a.model}_donor_postcast_v1' if a.postcast_calibration
            else f'{a.model}_donor_replication_v1')
    out=root/f'runs/mechanism_controls/{family}/{a.capability}/seed-{a.seed}'
    with training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{gpu}',blocking=False):
        if a.model=='dimoo':idle_gpu(out,gpu=gpu)
        elif not idle(gpu,[gpu]):raise ValueError(f'GPU{gpu} is not idle')
        try:
            job=job_inputs(root,a.model,a.capability,a.seed)
            generate(root,out,job,limit=1 if a.smoke else None,
                     start=a.shard_start or 0,end=a.shard_end,
                     postcast_calibration=a.postcast_calibration)
            if not a.smoke and not shard:score(root,out,job)
        except Exception as error:
            write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()))
            raise


if __name__=='__main__':main()
