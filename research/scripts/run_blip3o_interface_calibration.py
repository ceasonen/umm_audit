"""Complete-panel prompt calibration using the registered reference-core cache."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from skillbridge.mechanism_sequence import lock_json
from scripts.run_original_core_source_audit import write

NEW_ARMS = ('prompt_attribute', 'prompt_nuisance', 'natural_donor')
CACHED_ARMS = {'own': 'conditional_only', 'learned_donor': 'semantic_donor',
               'random_sign': 'random_sign'}


def read(path):
    return json.loads(path.read_text())


def load_design(root, seed, output):
    from skillbridge.matched_backbone_audit import reference_path
    from skillbridge.mechanism_completion import literal_count
    source = root/f'runs/mechanism_controls/semantic_interchange_native_v2/seed-{seed}'
    design = read(source/'design.json')
    core = reference_path(root, 'blip3o', 'counting', seed)
    if (design['core'] != str(core) or design['source_seed'] != seed or
            design.get('capability', 'counting') != 'counting' or len(design['pairs']) != 128 or
            read(source/'complete.json')['rows'] != 128):
        raise ValueError('complete registered reference-core counting cache required')
    if any(pair[role].get('capability')!='counting' for pair in design['pairs']
           for role in ('recipient','semantic_donor','nuisance_donor')):
        raise ValueError('complete registered reference-core counting cache required')
    if len({p['recipient']['split_key'] for p in design['pairs']})!=128:
        raise ValueError('duplicate registered recipient')
    for pair in design['pairs']:
        n,obj=literal_count(pair['recipient']['user_prompt'])
        nd,od=literal_count(pair['semantic_donor']['user_prompt'])
        nn,on=literal_count(pair['nuisance_donor']['user_prompt'])
        if n==nd or obj!=od or nn!=n or obj==on:
            raise ValueError('invalid registered attribute/nuisance pair')
    plan = dict(source=str(source), core=str(core), source_seed=seed, pairs=design['pairs'],
        new_arms=list(NEW_ARMS), cached_arms=CACHED_ARMS, rows=128,
        selection='all existing registered pairs; no output-dependent selection',
        prompt_contrast='frozen donor conditional minus frozen recipient conditional',
        budget='learned recipient conditional change; native squared-energy tolerance 20%',
        randomness='original decoder seed 8087 + registered pair index',
        natural_donor='both frozen donor branches, actual donor prompt; base capability check only',
        primary_comparisons=['prompt_attribute-minus-own', 'prompt_attribute-minus-prompt_nuisance',
                             'learned_donor-minus-own'],
        scope='same reference update; local carrier calibration, not natural-pathway identification')
    lock_json(output/'design.json', plan)
    return source, core, plan


def generate(root, seed, output):
    import torch
    from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o
    from skillbridge.interface_calibration import prompt_contrast_state, validate_generated_identity
    from skillbridge.matched_backbone_audit import validate_reference
    from skillbridge.mechanism_sequence import replace_condition
    from skillbridge.readout_probe import set_core_state
    source, core, design = load_design(root, seed, output)
    ck = torch.load(core, map_location='cpu', weights_only=True)
    sites = validate_reference(core, ck['metadata'], 'blip3o', 'counting', seed)
    full = ck['state_dict']
    if len(full) != sites or any(tuple(v.shape) != (64, 64) or not torch.isfinite(v).all() for v in full.values()):
        raise ValueError('reference core state differs')
    gain, suffixes = checkpoint_route_spec(core)
    write(output/'status.json', dict(status='loading_model', updated=time.time()))
    adapter, route, kwargs = load_blip3o(root, rank=64, timestep_buckets=1,
        method='tied_core', gain_limit=gain, target_suffixes=suffixes, task='generation')
    set_core_state(route, {k:torch.zeros_like(v) for k,v in full.items()})
    started = time.monotonic()
    bank = {a: [] for a in NEW_ARMS}
    for i, pair in enumerate(design['pairs']):
        unit = output/'units'/f'{i:05d}'
        marker = unit/'complete.json'
        if marker.exists():
            saved = read(marker)
            for arm in NEW_ARMS:
                row = saved['predictions'][arm]
                validate_generated_identity(row, pair, i, seed, arm)
                if not (output/arm/row['image']).is_file():
                    raise ValueError('resume image identity differs')
                bank[arm].append(row)
            continue
        cached = torch.load(source/'units'/f'{i:05d}'/'interfaces.pt', map_location='cpu', weights_only=True)
        if cached['recipient_zero'].dtype != adapter.diffusion_pipeline.dtype(adapter.diffusion_pipeline.unet):
            raise ValueError('native carrier precision differs from registered cache')
        states, gates = {}, {}
        for arm, role in (('prompt_attribute', 'semantic_donor'), ('prompt_nuisance', 'nuisance_donor')):
            states[arm], gates[arm] = prompt_contrast_state(cached['recipient_zero'],
                cached['recipient_full'], cached[f'{role}_zero'])
        states['natural_donor'] = cached['semantic_donor_zero'].clone()
        lock_json(unit/'activation-audit.json', gates)
        order = list(NEW_ARMS); random.Random(seed+i).shuffle(order)
        rows = {}
        for arm in order:
            actual_prompt = pair['semantic_donor' if arm=='natural_donor' else 'recipient']['user_prompt']
            adapter.reset_condition_cache()
            with replace_condition(adapter.diffusion_pipeline, states[arm], seed=8087+i,
                                   verify_native=(arm=='natural_donor' and i==0)):
                image = adapter.generate(actual_prompt, seed=8087+i, **kwargs)
            path = output/arm/'images'/f'{i:05d}.png';path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            row = dict(split_key=pair['recipient']['split_key'], prompt=actual_prompt,
                seed=8087+i, image=f'images/{i:05d}.png', condition=arm, pair_index=i,
                source_seed=seed, generation_prompt=actual_prompt)
            rows[arm] = row;bank[arm].append(row)
        write(marker, dict(pair_index=i, predictions=rows, gates=gates))
        write(output/'status.json', dict(status='generating', completed=i+1, total=128,
            elapsed=time.monotonic()-started, updated=time.time()))
        print(f'calibration {seed}: {i+1}/128', flush=True)
    for arm, rows in bank.items():
        (output/arm/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(output/'generation-complete.json', dict(rows=128, arms=list(NEW_ARMS), updated=time.time()))


def score(root, seed, output):
    from scripts.analyze_readout_evidence import endpoint, lines
    from skillbridge.mechanism_completion import attribute_outcome
    from skillbridge.interface_calibration import cached_readout_values, validate_generated_identity
    source, _, design = load_design(root, seed, output)
    if read(output/'generation-complete.json')['rows'] != 128:
        raise ValueError('all generated units required before scoring')
    scores = {}
    for arm, cached_arm in CACHED_ARMS.items():
        scores[arm] = {}
        for role in ('recipient', 'donor'):
            keys, values, _ = endpoint(source/'scored'/cached_arm/role, 'generation')
            scores[arm][role] = cached_readout_values(design['pairs'], keys, values, role)
    for arm in NEW_ARMS:
        scores[arm] = {}
        original = lines(output/arm/'predictions.jsonl')
        if len(original) != 128:
            raise ValueError('generated prediction identity differs')
        for i, (row, pair) in enumerate(zip(original, design['pairs'], strict=True)):
            validate_generated_identity(row, pair, i, seed, arm)
        for role in ('recipient', 'donor'):
            folder = output/'scored'/arm/role;folder.mkdir(parents=True, exist_ok=True)
            records, predictions = [], []
            for i,(pair,row) in enumerate(zip(design['pairs'],original,strict=True)):
                target = pair['recipient' if role=='recipient' else 'semantic_donor']
                key = f'{pair["recipient"]["split_key"]}::{role}'
                records.append(dict(target, split_key=key, benchmark_id=str(i)))
                predictions.append(dict(row, split_key=key, prompt=target['user_prompt'],
                    image=str((output/arm/row['image']).resolve())))
            for name,rows in (('records',records),('predictions',predictions)):
                path=folder/f'{name}.jsonl';text=''.join(json.dumps(r)+'\n' for r in rows)
                if path.exists() and path.read_text()!=text:raise ValueError('scoring design changed')
                path.write_text(text)
            stage=folder/'official-stage'
            if not stage.exists():
                subprocess.run([str(root/'environments/dimoo/.venv/bin/python'),'-P',str(root/'scripts/stage_oral_predictions.py'),
                    '--records',str(folder/'records.jsonl'),'--predictions',str(folder/'predictions.jsonl'),'--output',str(stage)],check=True)
            summary=stage/'t2i_compbench/official-summary.json'
            if not summary.exists():
                py=root/'environments/t2i_compbench/.venv/bin/python'
                subprocess.run([str(py),'-P',str(root/'scripts/run_t2i_compbench_official.py'),
                    '--repository',str(root/'external/T2I-CompBench'),'--lock',str(root/'configs/evaluators.lock.json'),
                    '--staged-root',str(stage/'t2i_compbench'),'--output',str(summary),'--python',str(py),
                    '--no-revision-verification'],check=True)
            keys,values,_=endpoint(folder,'generation')
            indexed={int(k[1]):float(v) for k,v in zip(keys,values,strict=True)}
            if set(indexed)!=set(range(128)):raise ValueError('complete official scores required')
            scores[arm][role]=[indexed[i] for i in range(128)]
    items, means = {}, {}
    for arm,roles in scores.items():
        items[arm]=[dict(pair_index=i,**attribute_outcome(r,d))
            for i,(r,d) in enumerate(zip(roles['recipient'],roles['donor'],strict=True))]
        means[arm]={key:100*sum(row[key] for row in items[arm])/128
            for key in ('object_present','recipient_count_hit','donor_count_hit')}
    gates={arm:[read(output/'units'/f'{i:05d}'/'activation-audit.json')[arm] for i in range(128)]
           for arm in ('prompt_attribute','prompt_nuisance')}
    report=dict(complete=True,source_seed=seed,rows=128,conditions=means,items=items,raw_scores=scores,
        norm_matching={arm:dict(matched_units=sum(g['passed'] for g in values),total_units=128,
            all_matched=all(g['passed'] for g in values)) for arm,values in gates.items()},
        contrasts={arm:{key:means[arm][key]-means['own'][key] for key in means[arm]} for arm in means if arm!='own'},
        specificity={key:means['prompt_attribute'][key]-means['prompt_nuisance'][key] for key in means['own']},
        scope=design['scope'],natural_donor_is_matched_control=False,semantic_sensitivity_assumed=False)
    write(output/'complete.json',report)
    write(output/'status.json',dict(status='complete',updated=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(17,29,43,71),required=True)
    p.add_argument('--gpu',type=int,choices=(1,2),required=True)
    p.add_argument('--mode',choices=('all','generate','score'),default='all')
    a=p.parse_args();root=a.root.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('physical GPU assignment differs')
    output=root/f'runs/mechanism_controls/blip3o_interface_calibration_v1/counting/seed-{a.seed}'
    if a.mode=='all':
        from skillbridge.training_overlap import training_lock
        with ExitStack() as stack:
            stack.enter_context(training_lock(root/f'runs/mechanism_controls/gpu_serial_slots/gpu-{a.gpu}',blocking=False))
            stack.enter_context(training_lock(output,blocking=False))
            free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if free<20000 or util>5:raise ValueError('calibration requires an idle GPU with 20 GB free')
            uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'],text=True)
            if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
                raise ValueError('calibration requires a GPU without another compute process')
            for mode in ('generate','score'):
                cmd=[sys.executable,'-P',str(Path(__file__).resolve()),'--root',str(root),
                     '--seed',str(a.seed),'--gpu',str(a.gpu),'--mode',mode]
                result=subprocess.run(cmd)
                if result.returncode:
                    write(output/'status.json',dict(status='failed',mode=mode,returncode=result.returncode,updated=time.time()))
                    raise SystemExit(result.returncode)
    elif a.mode=='generate':generate(root,a.seed,output)
    else:score(root,a.seed,output)


if __name__=='__main__':main()
