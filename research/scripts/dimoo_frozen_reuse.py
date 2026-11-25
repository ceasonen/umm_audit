"""Reuse the first complete DiMOO frozen panel after exact input/output checks."""
import json
import math
from pathlib import Path

from scripts.analyze_readout_evidence import endpoint
from scripts.run_uniworld_replication_endpoints import rows, write
from skillbridge.dimoo_reference_evaluation import campaign_plan, validate_metadata, validate_response


def read(path):
    return json.loads(path.read_text())


def prepare_reuse(base,out,capability,seed,u,g,inputs,*,decode,live_answers,live_image):
    if seed==17:return None
    if seed not in (29,43,71):raise ValueError('unregistered training repeat')
    provider=base.parent/'seed-17/endpoints'
    if not (provider/'complete.json').is_file():return None
    relatives=('understanding/tokens-32/zero','understanding/tokens-128/zero','generation/zero')
    destinations=[out/r for r in relatives]
    if (any(p.is_symlink() for p in destinations)
            and any(p.exists() and not p.is_symlink() for p in destinations)):
        raise ValueError('mixed reused and independent frozen panels require inspection')
    for relative in relatives:
        dest=out/relative
        if dest.is_symlink() and dest.resolve()!=(provider/relative).resolve():
            raise ValueError('frozen link names a different provider')
        if dest.exists() and not dest.is_symlink():return None
        if (provider/relative).is_symlink():raise ValueError('provider must contain its own inference')
    done=read(provider/'complete.json');design=read(provider/'design.json')
    if (done.get('complete') is not True or done.get('capability')!=capability or done.get('seed')!=17
            or read(provider/'status.json').get('status')!='complete'):
        raise ValueError('completed same-capability first run required')
    validate_metadata(design['metadata'],capability,17)
    expected=dict(understanding_records=u,generation_records=g,protocol=campaign_plan(),source_selection_uses_target=False)
    if any(design.get(k)!=v for k,v in expected.items()):raise ValueError('frozen input or protocol differs')
    parity=read(provider/'zero-bypass-parity.json')
    if (parity.get('answer_tokens_equal') is not True or parity.get('pixels_equal') is not True
            or parity['zero_answers']!=parity['bypass_answers']):
        raise ValueError('provider zero/bypass parity failed')
    if len(inputs)!=len(u) or not u or not g:raise ValueError('complete native inputs required')
    for i,prepared in enumerate(inputs):
        if prepared!=read(provider/'understanding-inputs'/f'{i:04d}'/'input.json'):
            raise ValueError('native crop, image tokens, or prompt tokens differ')
    for budget in (32,128):
        folder=provider/f'understanding/tokens-{budget}/zero'
        if len(list(folder.glob('*.json')))!=len(u):raise ValueError('frozen answer coverage differs')
        for i,record in enumerate(u):
            response=read(folder/f'{i:04d}.json')
            validate_response(record,response,state='zero',budget=budget,decode=decode)
            if i==0 and response['token_record']['raw_tokens']!=live_answers[str(budget)]:
                raise ValueError('live frozen answer tokens differ from provider')
    folder=provider/'generation/zero'
    if not (folder/'official-stage/staging_manifest.json').is_file():
        raise ValueError('complete frozen scoring stage required')
    predictions=rows(folder/'predictions.jsonl')
    if len(predictions)!=len(g):raise ValueError('complete frozen generation required')
    for i,(record,prediction) in enumerate(zip(g,predictions,strict=True)):
        expected=dict(split_key=record['split_key'],prompt=record['user_prompt'],condition='zero',
                      seed=8087+i,image=f'images/{i:05d}.png')
        if any(prediction.get(k)!=v for k,v in expected.items()):
            raise ValueError('frozen generation identity or noise differs')
        image=folder/prediction['image']
        if not image.is_file() or not image.stat().st_size:raise ValueError('frozen image missing')
    from PIL import Image
    from skillbridge.decoder_sanity import same_pixels
    with Image.open(folder/predictions[0]['image']) as saved:
        if not same_pixels(live_image,saved):raise ValueError('live frozen pixels differ from provider')
    _,scores,categories=endpoint(folder,'generation')
    index=rows(folder/'official-stage/index.jsonl');records={r['split_key']:r for r in g};items=[]
    for row,score,category in zip(index,scores,categories,strict=True):
        record=records[row['split_key']]
        if (row['benchmark']!=record['benchmark'] or str(row['benchmark_id'])!=str(record['benchmark_id'])
                or not math.isfinite(float(score)) or not 0<=score<=1):
            raise ValueError('frozen official scoring identity or value differs')
        items.append(dict(split_key=row['split_key'],benchmark_id=int(row['benchmark_id']),category=category,score=float(score)))
    if items!=done['generation']['zero']['items']:raise ValueError('frozen official raw scores differ')
    report=dict(provider=str(provider),provider_seed=17,consumer_seed=seed,capability=capability,
        reused_answers=2*len(u),reused_images=len(g),protocol=campaign_plan(),
        prepared_inputs_checked=len(inputs),live_answer_replays=2,live_image_replays=1,
        scope='Same frozen model, exact native inputs, budgets and paired noise; full-core inference stays independent.')
    dest=out/'baseline-reuse.json'
    if dest.exists() and read(dest)!=report:raise ValueError('existing reuse provenance differs')
    for relative in relatives:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_symlink():target.symlink_to(provider/relative,target_is_directory=True)
    write(dest,report)
    return report
