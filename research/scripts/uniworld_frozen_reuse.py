"""Reuse a completed same-capability frozen panel without rewriting its artifacts."""
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_provider(provider, capability, u, g, kwargs):
    from scripts.analyze_readout_evidence import endpoint
    from scripts.run_uniworld_replication_endpoints import validate_metadata
    from skillbridge.source_measurement import validate_completed_predictions

    done=read(provider/'complete.json')
    if (done.get('complete') is not True or done.get('capability') != capability
            or done.get('seed') != 17 or read(provider/'status.json').get('status') != 'complete'):
        raise ValueError('only the completed first run can provide a frozen baseline')
    design=read(provider/'design.json')
    validate_metadata(design['checkpoint_metadata'],capability,17)
    expected=dict(understanding_records=u,generation_records=g,answer_budgets=[32,128],
                  generation_kwargs=kwargs,inference_seed=8087,source_selection_uses_target=False)
    if any(design.get(k) != v for k,v in expected.items()):
        raise ValueError('frozen baseline inputs or native inference settings differ')
    if read(provider/'zero-bypass-parity.json') != dict(answer_equal=True,pixels_equal=True):
        raise ValueError('provider zero/bypass check did not pass')
    for budget in (32,128):
        folder=provider/f'understanding/tokens-{budget}/zero'
        if folder.is_symlink():raise ValueError('provider must contain its own inference')
        predictions=rows(folder/'predictions.jsonl')
        validate_completed_predictions(u,predictions)
        if any(p.get('condition') != 'zero' or p.get('max_new_tokens') != budget
               or not isinstance(p.get('prediction'),str) for p in predictions):
            raise ValueError('frozen answer condition or decoding budget differs')
        endpoint(folder,'understanding')
    folder=provider/'generation/zero'
    if folder.is_symlink():raise ValueError('provider must contain its own inference')
    if not (folder/'official-stage/staging_manifest.json').is_file():
        raise ValueError('completed frozen staging manifest required for read-only reuse')
    predictions=rows(folder/'predictions.jsonl')
    if len(predictions) != len(g):raise ValueError('complete frozen generation required')
    for i,(record,prediction) in enumerate(zip(g,predictions)):
        expected=dict(split_key=record['split_key'],prompt=record['user_prompt'],condition='zero',
                      seed=8087+i,image=f'images/{i:05d}.png')
        if any(prediction.get(k) != v for k,v in expected.items()):
            raise ValueError('frozen generation prompt, condition or noise differs')
        image=folder/prediction['image']
        if not image.is_file() or image.stat().st_size == 0:raise ValueError('frozen image missing')
    _,scores,categories=endpoint(folder,'generation')
    index=rows(folder/'official-stage/index.jsonl');by_id={r['split_key']:r for r in g}
    items=[]
    for row,score,category in zip(index,scores,categories,strict=True):
        record=by_id[row['split_key']]
        if (row['benchmark'] != record['benchmark'] or str(row['benchmark_id']) != str(record['benchmark_id'])
                or not math.isfinite(float(score)) or not 0 <= score <= 1):
            raise ValueError('frozen official scoring identity or value differs')
        items.append(dict(split_key=row['split_key'],benchmark_id=int(row['benchmark_id']),category=category,score=float(score)))
    if items != done['generation']['zero']['items']:
        raise ValueError('frozen raw scores differ from the completed provider')


def prepare_reuse(base,out,capability,seed,u,g,kwargs):
    """Return provenance if all three zero panels can be linked; preserve existing work."""
    if seed == 17:return None
    if seed not in (29,43,71):raise ValueError('unregistered training repeat')
    provider=base.parent/'seed-17/endpoints'
    if not (provider/'complete.json').is_file():return None
    relatives=['understanding/tokens-32/zero','understanding/tokens-128/zero','generation/zero']
    for relative in relatives:
        dest=out/relative
        if dest.is_symlink() and dest.resolve() != (provider/relative).resolve():
            raise ValueError('existing frozen link names another provider')
        if dest.exists() and not dest.is_symlink():return None
    validate_provider(provider,capability,u,g,kwargs)
    report=dict(provider=str(provider),provider_seed=17,consumer_seed=seed,capability=capability,
        reused_answers=2*len(u),reused_images=len(g),answer_budgets=[32,128],
        inference_seed=8087,generation_kwargs=kwargs,
        scope='Same frozen model, inputs, decoding and paired noise; full-core inference remains independent.')
    destination=out/'baseline-reuse.json'
    if destination.exists() and read(destination) != report:
        raise ValueError('existing frozen reuse provenance differs')
    for relative in relatives:
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.is_symlink():dest.symlink_to(provider/relative,target_is_directory=True)
    from scripts.run_uniworld_replication_endpoints import write
    write(destination,report)
    return report


def validate_zero_route(adapter,layers):
    """Establish the algebraic frozen path before reusing a frozen panel."""
    import torch
    if not layers or any(m.training for model in [adapter.model,*adapter.text_encoders] for m in model.modules()):
        raise ValueError('all frozen inference modules must be in evaluation mode')
    for layer in layers:
        zero_tensors=[layer.shared_core.weight,layer.capability_delta,
                      *[m.weight for m in layer.private_output.values()]]
        if (any(torch.count_nonzero(t).item() for t in zero_tensors)
                or layer._shared_modulation_transform is not None or layer._capability_scales):
            raise ValueError('route does not reduce to the frozen projection')
