#!/usr/bin/env python3
"""Validate completed reference-generation scores against inputs and staged images."""
import argparse
import json
from pathlib import Path
from statistics import mean

from PIL import Image

from scripts.analyze_readout_evidence import endpoint, lines


def read(path):
    return json.loads(path.read_text())


def audit_panel(folder, records, state, resolution, *, condition_key='condition', noise_seeds=None):
    if condition_key not in ('condition','intervention'):
        raise ValueError('unknown recorded condition field')
    predictions = lines(folder/'predictions.jsonl')
    if len(predictions)!=len(records):
        raise ValueError('generation input coverage differs')
    noises=list(range(8087,8087+len(records))) if noise_seeds is None else list(noise_seeds)
    if len(noises)!=len(records) or any(type(s) is not int for s in noises):
        raise ValueError('one recorded integer noise seed per input required')
    for i,(r,p) in enumerate(zip(records,predictions)):
        expected=dict(split_key=r['split_key'],prompt=r['user_prompt'],seed=noises[i])
        expected[condition_key]=state
        if any(p.get(k)!=v for k,v in expected.items()):
            raise ValueError('generation prompt, noise seed, or condition differs')
    keys, scores, categories = endpoint(folder, 'generation')
    stage = folder/'official-stage'
    index = lines(stage/'index.jsonl')
    by_id = {r['split_key']:(r,p) for r,p in zip(records,predictions)}
    items = []
    for row,key,score,category in zip(index,keys,scores,categories,strict=True):
        record,prediction = by_id[row['split_key']]
        if row['benchmark']!=record['benchmark'] or str(row['benchmark_id'])!=str(record['benchmark_id']):
            raise ValueError('official index refers to a different benchmark item')
        original = folder/prediction['image']; staged = stage/row['image']
        if not original.is_file() or not staged.is_file() or original.stat().st_size==0:
            raise ValueError('generated or scored image missing')
        # Staging uses hard links. A copied image is compared directly, without
        # computing a checksum, only when file identity cannot establish it.
        if not original.samefile(staged) and original.read_bytes()!=staged.read_bytes():
            raise ValueError('official evaluator scored a different image')
        with Image.open(original) as image:
            if image.size!=(resolution,resolution):
                raise ValueError('native generation resolution differs')
        if not 0<=float(score)<=1:
            raise ValueError('official score outside its declared unit range')
        items.append(dict(split_key=row['split_key'], benchmark_id=int(row['benchmark_id']),
                          category=category, score=float(score)))
    summary = read(stage/'t2i_compbench/official-summary.json')
    category_scores = {}
    for name in sorted(set(categories)):
        selected = [r['score'] for r in items if r['category']==name]
        value = mean(selected)
        if abs(value-summary['categories'][name]['score'])>1e-8:
            raise ValueError('category mean differs from official raw scores')
        category_scores[name] = dict(rows=len(selected), score_x100=100*value)
    return dict(rows=len(items), score_x100=100*mean(scores), categories=category_scores,
                official_scores_validated=True, staged_images_match=True, resolution=resolution, items=items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a=p.parse_args(); root=a.root.resolve()
    runs=[]
    for cap in ('counting','spatial'):
        filename='records.jsonl' if cap=='counting' else 'records-balanced-1000.jsonl'
        records=lines(root/f'data/oral_benchmarks_confirmatory_v1/{cap}/tiers/confirmatory/{filename}')
        if len(records)!=1000:
            raise ValueError('reference endpoint requires 1,000 fixed prompts')
        expected_categories={'numeracy':1000} if cap=='counting' else {'2d_spatial':500,'3d_spatial':500}
        for seed in (17,29,43,71):
            for backbone,cohort in [('blip3o','reference'),('janusflow','replication')]:
                core=root/f'runs/mechanism_controls/mechanism_sequence_v1/{backbone}/{cap}/seed-{seed}/{cohort}/core.pt'
                base=core.parent/'endpoints/base'
                design=read(base/'generation-design.json')
                kwargs = (dict(height=512,width=512,diffusion_steps=30,guidance_scale=3.0) if backbone=='blip3o'
                          else dict(image_size=384,steps=30,cfg_scale=5.0))
                expected=dict(core=str(core),checkpoint_method='tied_core',checkpoint_capability=cap,
                    test_capability=cap,backbone=backbone,source_seed=seed,generation_kwargs=kwargs,inference_seed=8087,
                    source_selection_uses_target=False,cache_policy='reset per image')
                if any(design.get(k)!=v for k,v in expected.items()):
                    raise ValueError('reference-generation design differs from native protocol')
                panels={s:audit_panel(base/'generation'/s,records,s,512 if backbone=='blip3o' else 384) for s in ('zero','full')}
                for panel in panels.values():
                    if {k:v['rows'] for k,v in panel['categories'].items()}!=expected_categories:
                        raise ValueError('unexpected category coverage')
                z,f=(panels[s] for s in ('zero','full'))
                if [(r['split_key'],r['category'],r['benchmark_id']) for r in z['items']]!=[(r['split_key'],r['category'],r['benchmark_id']) for r in f['items']]:
                    raise ValueError('full/frozen scoring indices are not paired')
                runs.append(dict(backbone=backbone,capability=cap,seed=seed,design=design,panels=panels,
                    delta_score_x100=f['score_x100']-z['score_x100'],
                    category_delta_score_x100={c:f['categories'][c]['score_x100']-z['categories'][c]['score_x100'] for c in expected_categories},
                    historical_completion_marker=read(base/'generation-complete.json')))
    aggregates={}
    for b in ('blip3o','janusflow'):
        aggregates[b]={}
        for cap in ('counting','spatial'):
            selected=[r for r in runs if r['backbone']==b and r['capability']==cap]
            aggregates[b][cap]=dict(states={s:dict(score_x100=mean(r['panels'][s]['score_x100'] for r in selected),
                categories={c:mean(r['panels'][s]['categories'][c]['score_x100'] for r in selected) for c in selected[0]['panels'][s]['categories']}) for s in ('zero','full')},
                delta_score_x100=mean(r['delta_score_x100'] for r in selected),
                run_deltas=[r['delta_score_x100'] for r in selected])
    print(json.dumps(dict(complete=True,panels=32,scored_images=32000,runs=runs,aggregates=aggregates,
        scope='Shared prompts and within-backbone paired noise; native generation resolution, decoder, CFG, and tokenization differ. Not architecture-only causality or cross-backbone donor validation.',
        completion_rule='Raw per-item scores, indices, generated images, and aggregate identities override stale generation-only markers.'),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
