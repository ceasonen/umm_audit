#!/usr/bin/env python3
"""Run the registered full Show-o2 native answer and image endpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_uniworld_replication_endpoints import rows, write, validate_records
from skillbridge.native_extension_endpoints import (endpoint_plan, validate_training_configuration,
    prepare_native_frozen_reuse)


def registered_gpu(gpu, environ):
    if gpu not in (2, 5) or environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Show-o2 endpoint GPU must be 2 or 5 and match CUDA visibility')
    return gpu


def run(root, capability, seed):
    import torch
    from scripts.run_blip3o_readout_followup import targets
    from scripts.audit_reference_generation import audit_panel
    from scripts.train_cross_backbone_pilot import attach_route
    from skillbridge.adapter_checkpoint import load_capability_checkpoint
    from skillbridge.cross_backbone_evaluation import generate_fixed_predictions, generate_understanding_predictions
    from skillbridge.decoder_sanity import bypass_layers, same_pixels
    from skillbridge.showo2_runtime import load_components, Showo2InferenceAdapter
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.sidecar import RoutedLowRankLinear
    from skillbridge.source_measurement import audit_pair, disjoint_records, validate_completed_predictions

    plan=endpoint_plan('showo2')
    base=root/f'runs/mechanism_controls/showo2_reference_replication_v1/{capability}/seed-{seed}'
    out=base/'endpoints'
    config=json.loads((base/'training-complete.json').read_text())['configuration']
    validate_training_configuration(config,'showo2',capability,seed)
    u=rows(root/f'data/route_transfer_oral_v2/capabilities/{capability}/understanding/test/records.jsonl')
    g_path=targets(root,capability);g=rows(g_path)
    validate_records(u,g,capability)
    disjoint_records(rows(base/'training-inputs.jsonl'),u)
    checkpoint=torch.load(base/'core.pt',map_location='cpu',weights_only=True)
    md=checkpoint['metadata'];full=checkpoint['state_dict']
    expected=dict(backbone='showo2',capability=capability,seed=seed,method='tied_core',
                  rank=64,timestep_buckets=1,gain_limit=.2 if capability=='counting' else .1)
    if any(md.get(k)!=v for k,v in expected.items()) or not full or \
            any(tuple(v.shape)!=(64,64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('Show-o2 trained checkpoint identity or shape differs')
    lock_json(out/'design.json',dict(protocol=plan,metadata=md,source_selection_uses_target=False,
                                    understanding_records=u,generation_records=g))
    write(out/'status.json',dict(status='loading',updated=time.time()))
    model,tokenizer,ids,vae=load_components(root)
    route=model.showo.model
    attach_route(route,rank=64,timestep_buckets=1,method='tied_core',
                 gain_limit=expected['gain_limit'],target_suffixes=('q_proj','v_proj'))
    load_capability_checkpoint(route,base/'core.pt')
    layers=[x for x in route.modules() if isinstance(x,RoutedLowRankLinear)]
    if len(layers)!=len(full):raise ValueError('native route site coverage differs')
    adapter=Showo2InferenceAdapter(model,tokenizer,ids,vae,route)
    states=dict(zero={k:torch.zeros_like(v) for k,v in full.items()},full=full)
    set_core_state(route,states['zero'])
    write(out/'status.json',dict(status='zero_bypass',updated=time.time()))
    answer=adapter.understand(u[0]['user_image'],u[0]['user_prompt'],max_new_tokens=32)
    image=adapter.generate(g[0]['user_prompt'],seed=plan['inference_seed'],**plan['generation_kwargs'])
    with bypass_layers(layers):
        bypass_answer=adapter.understand(u[0]['user_image'],u[0]['user_prompt'],max_new_tokens=32)
        bypass_image=adapter.generate(g[0]['user_prompt'],seed=plan['inference_seed'],**plan['generation_kwargs'])
    parity=dict(answer_equal=answer==bypass_answer,pixels_equal=same_pixels(image,bypass_image))
    write(out/'zero-bypass-parity.json',parity)
    if not all(parity.values()):raise ValueError('Show-o2 native zero/bypass parity failed')
    reuse=prepare_native_frozen_reuse(base,out,'showo2',capability,seed,u,g,answer,image)
    reports={}
    for budget in plan['answer_budgets']:
        for state in plan['states']:
            set_core_state(route,states[state]);dest=out/'understanding'/f'tokens-{budget}'/state
            write(out/'status.json',dict(status='understanding',budget=budget,state=state,updated=time.time()))
            if not (reuse and state=='zero'):
                generate_understanding_predictions(adapter,u,dest,understanding_kwargs={'max_new_tokens':budget},
                    metadata=dict(condition=state,max_new_tokens=budget),resume=True)
            validate_completed_predictions(u,rows(dest/'predictions.jsonl'))
        reports[str(budget)]=audit_pair(rows(out/'understanding'/f'tokens-{budget}'/'zero/predictions.jsonl'),
                                        rows(out/'understanding'/f'tokens-{budget}'/'full/predictions.jsonl'))
    write(out/'understanding-report.json',reports)
    for state in plan['states']:
        set_core_state(route,states[state]);dest=out/'generation'/state
        write(out/'status.json',dict(status='generation',state=state,updated=time.time()))
        if not (reuse and state=='zero'):
            generate_fixed_predictions(adapter,g,dest,seed=plan['inference_seed'],
                generation_kwargs=plan['generation_kwargs'],metadata=dict(condition=state),resume=True)
    write(out/'inference-complete.json',dict(understanding_answers=len(u)*4,
          generation_images=2*len(g),zero_bypass_parity=parity,frozen_reuse=reuse,
          evaluated_answers=2*len(u) if reuse else 4*len(u),
          generated_images=len(g) if reuse else 2*len(g)))
    del model,adapter,route,vae;gc.collect();torch.cuda.empty_cache()
    write(out/'status.json',dict(status='official_scoring',updated=time.time()))
    scorer=root/'environments/t2i_compbench/.venv/bin/python'
    for state in plan['states']:
        dest=out/'generation'/state;stage=dest/'official-stage'
        if not (stage/'staging_manifest.json').exists():
            subprocess.run([str(root/'environments/showo2/.venv/bin/python'),'-P',
                str(root/'scripts/stage_oral_predictions.py'),'--records',str(g_path),
                '--predictions',str(dest/'predictions.jsonl'),'--output',str(stage)],check=True)
        summary=stage/'t2i_compbench/official-summary.json'
        if not summary.exists():
            subprocess.run([str(scorer),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                '--repository',str(root/'external/T2I-CompBench'),'--lock',str(root/'configs/evaluators.lock.json'),
                '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),
                '--python',str(scorer),'--no-revision-verification'],check=True)
    panels={state:audit_panel(out/'generation'/state,g,state,plan['image_size']) for state in plan['states']}
    lock_json(out/'complete.json',dict(complete=True,backbone='showo2',capability=capability,seed=seed,
        understanding=reports,generation=panels,
        generation_delta_x100=panels['full']['score_x100']-panels['zero']['score_x100']))
    write(out/'status.json',dict(status='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capability',choices=('counting','spatial'),required=True)
    p.add_argument('--seed',type=int,choices=(17,29,43,71),required=True)
    p.add_argument('--gpu',type=int,choices=(2,5),default=5)
    a=p.parse_args()
    gpu=registered_gpu(a.gpu,os.environ)
    from skillbridge.training_overlap import training_lock
    from scripts.run_extension_campaign import idle
    root=a.root.resolve()
    out=root/f'runs/mechanism_controls/showo2_reference_replication_v1/{a.capability}/seed-{a.seed}/endpoints'
    with training_lock(out/'endpoint-job-lock',blocking=True):
        if (out/'complete.json').exists():return
        with training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{gpu}',blocking=False):
            if not idle(gpu,[gpu]):raise ValueError(f'GPU{gpu} is not idle')
            try:run(root,a.capability,a.seed)
            except Exception as error:
                write(out/'status.json',dict(status='failed',error=str(error),updated=time.time()))
                raise


if __name__=='__main__':main()
