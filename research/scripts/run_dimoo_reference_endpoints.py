#!/usr/bin/env python3
"""Evaluate every new DiMOO reference core with recorded native inputs and outputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from skillbridge.dimoo_reference_evaluation import (campaign_plan,validate_metadata,build_query,
    response_record,validate_response,paired_answers)
from skillbridge.mechanism_sequence import lock_json
from scripts.run_uniworld_replication_endpoints import rows,write,validate_records


def registered_endpoint_gpu(gpu, environ):
    if gpu not in (1, 3, 6, 7) or environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('endpoint GPU must be authorized and match CUDA visibility')
    return gpu


def completed_endpoint(out, capability, seed):
    marker = out / 'complete.json'
    if not marker.exists():
        return False
    report = json.loads(marker.read_text())
    if report.get('complete') is not True or report.get('capability') != capability or report.get('seed') != seed:
        raise ValueError('completed endpoint identity differs from requested job')
    return True


def prepare_input(adapter,record,index,out):
    import torch
    from skillbridge.dimoo_inference import _load_rgb_image,_vqvae_autocast
    from skillbridge.dimoo_replication import preprocessing_rng
    from skillbridge.decoder_sanity import same_pixels
    image_utils=adapter._import('utils.image_utils')
    image=_load_rgb_image(Path(record['user_image']))
    with preprocessing_rng(20260924+index):
        sizes=image_utils.generate_crop_size_list((256//32)**2,32)
        image=image_utils.var_center_crop(image,crop_size_list=sizes)
    folder=out/'understanding-inputs'/f'{index:04d}';folder.mkdir(parents=True,exist_ok=True)
    image_path=folder/'input.png'
    if image_path.exists():
        from PIL import Image
        with Image.open(image_path) as old:
            if not same_pixels(image,old):raise ValueError('saved crop differs')
    else:image.save(image_path)
    with torch.inference_mode(),_vqvae_autocast(adapter.vqvae):
        image_tokens=image_utils.encode_img_with_breaks(image,adapter.vqvae)
    prompt=adapter._import('utils.prompt_utils').generate_multimodal_understanding_prompt(record['user_prompt'])
    prompt_tokens=adapter.tokenizer(prompt)['input_ids']
    build_query(prompt_tokens,image_tokens,32)
    data=dict(record=record,crop_seed=20260924+index,image_size=list(image.size),prompt=prompt,
        prompt_tokens=prompt_tokens,image_tokens=image_tokens)
    lock_json(folder/'input.json',data)
    return data


def decode_answer(adapter,route,prepared,budget):
    import torch
    from skillbridge.routing import forward_step_routing
    from skillbridge.dimoo_understanding_contract import generate_single_block
    ids,start=build_query(prepared['prompt_tokens'],prepared['image_tokens'],budget)
    x=torch.tensor([ids],device=next(adapter.model.parameters()).device)
    with torch.inference_mode(),forward_step_routing(adapter.model,route,task='understanding',total_steps=budget):
        generated=generate_single_block(adapter.model,x,code_start=start,generation_length=budget,steps=budget)
    if not torch.equal(x[:,:start],generated[:,:start]) or not torch.equal(x[:,-1:],generated[:,-1:]):
        raise ValueError('inference modified fixed input or ending')
    return generated[0,start:-1].tolist()


def infer(root,base,out,capability,seed,u,g):
    import torch
    from scripts.evaluate_cross_backbone_generation import load_dimoo
    from skillbridge.adapter_checkpoint import load_capability_checkpoint
    from skillbridge.sidecar import RoutedLowRankLinear
    from skillbridge.readout_probe import set_core_state
    from skillbridge.decoder_sanity import bypass_layers,same_pixels
    from skillbridge.cross_backbone_evaluation import generate_fixed_predictions
    ck=torch.load(base/'core.pt',map_location='cpu',weights_only=True)
    validate_metadata(ck['metadata'],capability,seed);full=ck['state_dict']
    if not full or any(tuple(v.shape)!=(64,64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('reference core tensor shape or finiteness differs')
    adapter,route,kwargs=load_dimoo(root,rank=64,timestep_buckets=1,method='tied_core',
        gain_limit=.2 if capability=='counting' else .1,target_suffixes=('q_proj','v_proj'))
    adapter.model.eval();load_capability_checkpoint(route,base/'core.pt')
    if kwargs!=campaign_plan()['generation_kwargs']:raise ValueError('native generation configuration differs')
    lock_json(out/'design.json',dict(core=str(base/'core.pt'),metadata=ck['metadata'],
        understanding_records=u,generation_records=g,protocol=campaign_plan(),source_selection_uses_target=False))
    layers=[m for m in route.modules() if isinstance(m,RoutedLowRankLinear)]
    if len(layers)!=len(full):raise ValueError('checkpoint and route sites differ')
    states=dict(zero={k:torch.zeros_like(v) for k,v in full.items()},full=full)
    inputs=[]
    for i,record in enumerate(u):
        write(out/'status.json',dict(status='preparing_understanding_inputs',index=i,total=len(u),updated=time.time()))
        inputs.append(prepare_input(adapter,record,i,out))
    set_core_state(route,states['zero'])
    raw_answers={str(b):decode_answer(adapter,route,inputs[0],b) for b in (32,128)}
    image=adapter.generate(g[0]['user_prompt'],seed=8087,**kwargs)
    with bypass_layers(layers):
        bypass_answers={str(b):decode_answer(adapter,route,inputs[0],b) for b in (32,128)}
        bypass_image=adapter.generate(g[0]['user_prompt'],seed=8087,**kwargs)
    parity=dict(answer_tokens_equal=raw_answers==bypass_answers,pixels_equal=same_pixels(image,bypass_image),
        zero_answers=raw_answers,bypass_answers=bypass_answers)
    write(out/'zero-bypass-parity.json',parity)
    image.save(out/'zero-parity.png');bypass_image.save(out/'bypass-parity.png')
    if not parity['answer_tokens_equal'] or not parity['pixels_equal']:raise ValueError('zero core and native bypass differ')
    decode=lambda tokens:adapter.tokenizer.batch_decode(torch.tensor([tokens],dtype=torch.long),skip_special_tokens=True)[0]
    reuse=None
    if seed!=17:
        from types import SimpleNamespace
        from scripts.uniworld_frozen_reuse import validate_zero_route
        from scripts.dimoo_frozen_reuse import prepare_reuse
        validate_zero_route(SimpleNamespace(model=adapter.model,text_encoders=[adapter.vqvae]),layers)
        reuse=prepare_reuse(base,out,capability,seed,u,g,inputs,
                            decode=decode,live_answers=raw_answers,live_image=image)
    del image,bypass_image
    reports={}
    for budget in (32,128):
        panels={}
        for state in ('zero','full'):
            set_core_state(route,states[state]);panel=[]
            for i,(record,prepared) in enumerate(zip(u,inputs)):
                dest=out/'understanding'/f'tokens-{budget}'/state/f'{i:04d}.json'
                if dest.exists():
                    row=json.loads(dest.read_text());validate_response(record,row,state=state,budget=budget,decode=decode)
                else:
                    write(out/'status.json',dict(status='understanding',budget=budget,state=state,index=i,total=len(u),updated=time.time()))
                    tokens=decode_answer(adapter,route,prepared,budget)
                    row=response_record(record,state=state,budget=budget,tokens=tokens,decode=decode)
                    lock_json(dest,row)
                panel.append(row)
            panels[state]=panel
        reports[str(budget)]=paired_answers(u,panels['zero'],panels['full'],budget=budget,decode=decode)
        write(out/'understanding-report.json',reports)
    for state in ('zero','full'):
        set_core_state(route,states[state]);write(out/'status.json',dict(status='generation',state=state,updated=time.time()))
        if not (reuse and state=='zero'):
            generate_fixed_predictions(adapter,g,out/'generation'/state,seed=8087,generation_kwargs=kwargs,
                                       metadata=dict(condition=state),resume=True)
    write(out/'inference-complete.json',dict(understanding_answers=len(u)*4,generation_images=2000,
        zero_bypass_parity=parity,frozen_reuse=reuse,
        reused_answers=2*len(u) if reuse else 0,reused_images=len(g) if reuse else 0,
        evaluated_answers=2*len(u) if reuse else 4*len(u),
        generated_images=len(g) if reuse else 2*len(g)))


def preflight(root):
    from scripts.run_blip3o_readout_followup import targets
    from skillbridge.source_measurement import disjoint_records
    base=root/'runs/mechanism_controls/dimoo_reference_replication_v1'
    design=json.loads((base/'design.json').read_text());report=[]
    for job in design['jobs']:
        cap=job['capability'];u=rows(root/f'data/route_transfer_oral_v2/capabilities/{cap}/understanding/test/records.jsonl')
        g=rows(targets(root,cap));validate_records(u,g,cap)
        if any(not Path(r['user_image']).is_file() for r in u):raise ValueError('held-out image missing')
        disjoint_records(rows(Path(job['source_records'])),u)
        report.append(dict(capability=cap,seed=job['seed'],questions=len(u),generation_prompts=len(g)))
    expected={(j['capability'],j['seed']) for j in campaign_plan()['jobs']}
    if len(report)!=8 or {(r['capability'],r['seed']) for r in report}!=expected:raise ValueError('training job coverage differs')
    lock_json(base/'endpoint-plan.json',campaign_plan())
    write(base/'endpoint-preflight.json',dict(complete=True,jobs=report,understanding_answers=24576,
        generation_images=16000,checkpoints_complete=False,scope='input and protocol preflight before pending training'))


def run(root,capability,seed):
    from scripts.run_dimoo_reference_replication import validate_training
    from scripts.run_blip3o_readout_followup import targets
    from scripts.audit_reference_generation import audit_panel
    from skillbridge.source_measurement import disjoint_records
    base=root/f'runs/mechanism_controls/dimoo_reference_replication_v1/{capability}/seed-{seed}';out=base/'endpoints'
    design=json.loads((base.parent.parent/'design.json').read_text())
    job=next(j for j in design['jobs'] if j['capability']==capability and j['seed']==seed)
    if not validate_training(job):raise ValueError('full reference training incomplete')
    u=rows(root/f'data/route_transfer_oral_v2/capabilities/{capability}/understanding/test/records.jsonl')
    g_path=targets(root,capability);g=rows(g_path);validate_records(u,g,capability)
    disjoint_records(rows(base/'training-inputs.jsonl'),u)
    infer(root,base,out,capability,seed,u,g)
    import torch
    gc.collect();torch.cuda.empty_cache()
    write(out/'status.json',dict(status='official_scoring',updated=time.time()))
    scorer=root/'environments/t2i_compbench/.venv/bin/python'
    for state in ('zero','full'):
        dest=out/'generation'/state;stage=dest/'official-stage'
        if not (stage/'staging_manifest.json').exists():
            subprocess.run([str(root/'environments/dimoo/.venv/bin/python'),'-P',str(root/'scripts/stage_oral_predictions.py'),
                '--records',str(g_path),'--predictions',str(dest/'predictions.jsonl'),'--output',str(stage)],check=True)
        summary=stage/'t2i_compbench/official-summary.json'
        if not summary.exists():
            subprocess.run([str(scorer),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                '--repository',str(root/'external/T2I-CompBench'),'--lock',str(root/'configs/evaluators.lock.json'),
                '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),'--python',str(scorer),
                '--no-revision-verification'],check=True)
    panels={s:audit_panel(out/'generation'/s,g,s,512) for s in ('zero','full')}
    lock_json(out/'complete.json',dict(complete=True,capability=capability,seed=seed,
        understanding=json.loads((out/'understanding-report.json').read_text()),generation=panels,
        generation_delta_x100=panels['full']['score_x100']-panels['zero']['score_x100']))
    write(out/'status.json',dict(status='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--capability',choices=('counting','spatial'))
    p.add_argument('--seed',type=int,choices=(17,29,43,71))
    p.add_argument('--gpu',type=int,default=1,choices=(1,3,6,7))
    a=p.parse_args();root=a.root.resolve()
    if a.preflight_only:preflight(root);return
    if a.capability is None or a.seed is None:p.error('capability and seed required')
    gpu=registered_endpoint_gpu(a.gpu, os.environ)
    from skillbridge.training_overlap import training_lock
    from scripts.run_dimoo_reference_replication import idle_gpu
    out=root/f'runs/mechanism_controls/dimoo_reference_replication_v1/{a.capability}/seed-{a.seed}/endpoints'
    with training_lock(out/'endpoint-job-lock',blocking=True):
        if completed_endpoint(out,a.capability,a.seed):
            return
        with training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{gpu}',blocking=False):
            try:idle_gpu(out,gpu=gpu);run(root,a.capability,a.seed)
            except Exception as error:
                write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()));raise


if __name__=='__main__':main()
