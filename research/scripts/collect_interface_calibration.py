"""Independently reconstruct same-carrier calibration from official item scores."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from scripts.analyze_readout_evidence import endpoint, lines
from scripts.run_blip3o_interface_calibration import CACHED_ARMS, NEW_ARMS
from skillbridge.interface_calibration import cached_readout_values, validate_generated_identity
from skillbridge.mechanism_completion import attribute_outcome

SEEDS=(17,29,43,71)
METRICS=('object_present','recipient_count_hit','donor_count_hit')


def same_gate(saved, actual):
    """Float32 reduction order can change low bits; gate decisions stay exact."""
    if saved.keys()!=actual.keys():return False
    return all(math.isclose(v,actual[k],rel_tol=2e-6,abs_tol=1e-5)
               if type(v) in (int,float) and k!='tolerance' else v==actual[k]
               for k,v in saved.items())


def recorded_native_cast(zero, delta, saved, actual):
    """Reconstruct the cast that generation used, retaining its recorded scale."""
    state=(zero[0].float()+saved['scale']*delta).to(zero.dtype)
    realized=(state.float()-zero[0].float()).square().sum().item()
    actual=dict(actual,realized_energy=realized,relative_energy=realized/actual['reference_energy'])
    actual['passed']=.8<=actual['relative_energy']<=1.2
    return actual


def aggregate(reports):
    if len(reports)!=4 or {r['source_seed'] for r in reports}!=set(SEEDS) or any(
            not r.get('complete') or r['rows']!=128 for r in reports):
        raise ValueError('all four complete registered runs required')
    if reports[0].get('input_pairs') is None or any(
            r.get('input_pairs')!=reports[0]['input_pairs'] for r in reports):
        raise ValueError('all four runs require the same registered panel')
    for r in reports:
        if any(not g['all_matched'] or g['matched_units']!=128 or g['total_units']!=128
               for g in r['norm_matching'].values()):
            raise ValueError('every native budget gate must pass for calibration')
    arms=tuple(CACHED_ARMS)+NEW_ARMS
    conditions={a:{m:mean(r['conditions'][a][m] for r in reports) for m in METRICS} for a in arms}
    deltas={a:{m:[r['conditions'][a][m]-r['conditions']['own'][m] for r in reports]
               for m in METRICS} for a in arms if a!='own'}
    return dict(complete=True,runs=4,pairs_per_run=128,conditions=conditions,
        effects={a:{m:mean(v) for m,v in ms.items()} for a,ms in deltas.items()},
        ranges={a:{m:[min(v),max(v)] for m,v in ms.items()} for a,ms in deltas.items()},
        run_effects=deltas,
        specificity={m:mean(r['conditions']['prompt_attribute'][m]-
                            r['conditions']['prompt_nuisance'][m] for r in reports) for m in METRICS},
        specificity_by_run={m:[r['conditions']['prompt_attribute'][m]-
                              r['conditions']['prompt_nuisance'][m] for r in reports] for m in METRICS},
        natural_donor_is_matched_control=False,
        natural_donor_scope='both frozen donor branches with actual donor prompt; base capability check',
        scope='BLIP3-o counting; prompt contrasts use learned budgets; no natural-pathway claim')


def read(path):
    return json.loads(path.read_text())


def verify_run(root,seed,verify_carriers=False):
    if verify_carriers:
        import torch
        torch.set_num_threads(2)
    source=root/f'runs/mechanism_controls/semantic_interchange_native_v2/seed-{seed}'
    output=root/f'runs/mechanism_controls/blip3o_interface_calibration_v1/counting/seed-{seed}'
    plan=read(output/'design.json');original=read(source/'design.json');pairs=plan['pairs']
    if plan['source_seed']!=seed or plan['core']!=original['core'] or pairs!=original['pairs']:
        raise ValueError('same-update registered input identity differs')
    if len(pairs)!=128 or read(output/'generation-complete.json')['rows']!=128:
        raise ValueError('incomplete registered generation')
    cached={}
    for arm,old in CACHED_ARMS.items():
        cached[arm]={}
        for role in ('recipient','donor'):
            folder=source/'scored'/old/role
            rows=lines(folder/'predictions.jsonl')
            by_key={r['split_key']:r for r in rows}
            if len(by_key)!=128 or set(by_key)!={p['recipient']['split_key'] for p in pairs}:
                raise ValueError('cached registered identity differs')
            for i,p in enumerate(pairs):
                row=by_key[p['recipient']['split_key']]
                expected=p['recipient' if role=='recipient' else 'semantic_donor']['user_prompt']
                if (row['condition']!=old or row['seed']!=8087+i or row['prompt']!=expected or
                    row['generation_prompt']!=p['recipient']['user_prompt']):
                    raise ValueError('cached condition/prompt/noise differs')
            keys,values,_=endpoint(folder,'generation')
            cached[arm][role]=cached_readout_values(pairs,keys,values,role)
    scores=dict(cached)
    for arm in NEW_ARMS:
        rows=lines(output/arm/'predictions.jsonl')
        if len(rows)!=128:raise ValueError('incomplete prediction family')
        for i,(row,pair) in enumerate(zip(rows,pairs,strict=True)):
            validate_generated_identity(row,pair,i,seed,arm)
            marker=read(output/'units'/f'{i:05d}'/'complete.json')
            if marker['pair_index']!=i or marker['predictions'][arm]!=row:
                raise ValueError('generated unit marker differs')
        scores[arm]={}
        for role in ('recipient','donor'):
            folder=output/'scored'/arm/role
            predictions=lines(folder/'predictions.jsonl')
            expected_records=[dict(p['recipient' if role=='recipient' else 'semantic_donor'],
                split_key=p['recipient']['split_key']+'::'+role,benchmark_id=str(i)) for i,p in enumerate(pairs)]
            if lines(folder/'records.jsonl')!=expected_records or len(predictions)!=128:
                raise ValueError('readout target identity differs')
            for i,(pred,raw,rec) in enumerate(zip(predictions,rows,expected_records,strict=True)):
                expected=dict(raw,split_key=rec['split_key'],prompt=rec['user_prompt'])
                if any(pred.get(k)!=v for k,v in expected.items() if k!='image') or not pred['image'].endswith(
                        f'/{arm}/images/{i:05d}.png'):
                    raise ValueError('scored prediction provenance differs')
            keys,values,_=endpoint(folder,'generation')
            indexed={int(k[1]):(float(v),k) for k,v in zip(keys,values,strict=True)}
            if len(indexed)!=128 or set(indexed)!=set(range(128)):
                raise ValueError('official score coverage differs')
            for i,rec in enumerate(expected_records):
                if indexed[i][1][2:]!=(rec['user_prompt'],8087+i):
                    raise ValueError('official prompt/noise differs')
            scores[arm][role]=[indexed[i][0] for i in range(128)]
    gates={a:[] for a in ('prompt_attribute','prompt_nuisance')}
    for i in range(128):
        unit=output/'units'/f'{i:05d}';audit=read(unit/'activation-audit.json')
        learned_audit=read(source/'units'/f'{i:05d}'/'activation-audit.json')['semantic_donor']
        if (not learned_audit['passed'] or learned_audit['donor_energy']<=0 or
            not .8<=learned_audit['relative_energy']<=1.2 or
            abs(learned_audit['relative_energy']-learned_audit['realized_energy']/
                learned_audit['reference_energy'])>1e-10):
            raise ValueError('learned donor native budget gate differs')
        if verify_carriers:
            import torch
            from skillbridge.interface_calibration import prompt_contrast_state
            cache=torch.load(source/'units'/f'{i:05d}'/'interfaces.pt',map_location='cpu',weights_only=True)
            from skillbridge.mechanism_sequence import condition_interventions
            _,reconstructed=condition_interventions(cache['recipient_zero'],cache['recipient_full'],
                cache['semantic_donor_zero'],cache['semantic_donor_full'],
                cache['nuisance_donor_zero'],cache['nuisance_donor_full'],seed=8087+i)
            learned_actual=recorded_native_cast(cache['recipient_zero'],
                cache['semantic_donor_full'][0].float()-cache['semantic_donor_zero'][0].float(),
                learned_audit,reconstructed['semantic_donor'])
            if not same_gate(learned_audit,learned_actual):
                raise ValueError('native tensors do not reproduce learned donor gate')
        for arm,role in (('prompt_attribute','semantic_donor'),('prompt_nuisance','nuisance_donor')):
            g=audit[arm]
            if (g['reference_energy']<=0 or g['prompt_energy']<=0 or not g['unconditional_preserved'] or
                g['tolerance']!=.2 or abs(g['relative_energy']-g['realized_energy']/g['reference_energy'])>1e-10 or
                g['passed']!=(.8<=g['relative_energy']<=1.2)):
                raise ValueError('invalid realized native gate')
            if verify_carriers:
                _,actual=prompt_contrast_state(cache['recipient_zero'],cache['recipient_full'],cache[f'{role}_zero'])
                actual=recorded_native_cast(cache['recipient_zero'],
                    cache[f'{role}_zero'][0].float()-cache['recipient_zero'][0].float(),g,actual)
                if not same_gate(g,actual):raise ValueError('native tensors do not reproduce calibration gate')
            if read(unit/'complete.json')['gates'][arm]!=g:raise ValueError('unit gate differs')
            gates[arm].append(g)
    items={a:[dict(pair_index=i,**attribute_outcome(r,d)) for i,(r,d) in enumerate(
        zip(v['recipient'],v['donor'],strict=True))] for a,v in scores.items()}
    conditions={a:{m:100*mean(row[m] for row in rows) for m in METRICS} for a,rows in items.items()}
    saved=read(output/'complete.json')
    for key,value in (('raw_scores',scores),('items',items),('conditions',conditions)):
        if saved[key]!=value:raise ValueError(f'raw readouts do not reproduce {key}')
    norms={a:dict(matched_units=sum(g['passed'] for g in gs),total_units=128,
                  all_matched=all(g['passed'] for g in gs)) for a,gs in gates.items()}
    if saved['norm_matching']!=norms:raise ValueError('norm summary differs')
    return dict(complete=True,source_seed=seed,rows=128,input_pairs=pairs,conditions=conditions,norm_matching=norms,
                native_tensor_gates_recomputed=verify_carriers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-carriers',action='store_true')
    a=p.parse_args();reports=[verify_run(a.root,s,a.verify_carriers) for s in SEEDS]
    result=aggregate(reports);result['run_reports']=reports;result['official_panels_recomputed']=48
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(effects=result['effects'],specificity=result['specificity']),indent=2))


if __name__=='__main__':main()
