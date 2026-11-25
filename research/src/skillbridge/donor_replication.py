"""Paired attribute evaluation, retaining all registered donor outcomes."""
from statistics import mean

from skillbridge.mechanism_completion import literal_count,attribute_outcome

CONDITIONS=('zero','full','interface_replay','conditional_only','unconditional_only',
            'semantic_donor','nuisance_donor','random_sign')
SEEDS=(17,29,43,71)


def scoring_view(pairs,predictions,source_root,*,criterion,condition,source_seed,noises,capability):
    if criterion not in ('recipient','donor') or condition not in CONDITIONS:
        raise ValueError('unknown scoring criterion/condition')
    if capability not in ('counting','spatial'):
        raise ValueError('unknown capability')
    if not pairs or len(pairs)!=len(predictions) or len(noises)!=len(pairs):
        raise ValueError('complete paired input coverage required')
    if len({p['recipient']['split_key'] for p in pairs})!=len(pairs):
        raise ValueError('duplicate recipient')
    records=[];outputs=[];root=source_root.resolve()
    for i,(pair,prediction,noise) in enumerate(zip(pairs,predictions,noises,strict=True)):
        recipient=pair['recipient'];donor=pair['semantic_donor'];nuisance=pair['nuisance_donor']
        if capability=='counting':
            n,o=literal_count(recipient['user_prompt']);nd,od=literal_count(donor['user_prompt'])
            nn,on=literal_count(nuisance['user_prompt'])
            if n==nd or o!=od or n!=nn or o==on:raise ValueError('invalid count donor pair')
        elif any(r['benchmark']!='t2i_compbench_2d' for r in (recipient,donor,nuisance)):
            raise ValueError('registered donor spatial panel requires 2D relation targets')
        expected=dict(split_key=recipient['split_key'],prompt=recipient['user_prompt'],
                      seed=noise,condition=condition,source_seed=source_seed)
        if any(prediction.get(k)!=v for k,v in expected.items()):raise ValueError('generation identity differs')
        image=(root/prediction['image']).resolve()
        if root not in image.parents or not image.is_file():raise ValueError('missing or out-of-scope image')
        target=recipient if criterion=='recipient' else donor
        key=f"{recipient['split_key']}::{criterion}"
        records.append(dict(target,split_key=key,benchmark_id=str(i)))
        outputs.append(dict(prediction,split_key=key,prompt=target['user_prompt'],image=str(image),
            generation_prompt=recipient['user_prompt'],generation_split_key=recipient['split_key'],
            scoring_criterion=criterion,target_split_key=target['split_key']))
    return records,outputs


def summarize_job(capability,source_seed,panels,norm_flags):
    if set(panels)!=set(CONDITIONS):raise ValueError('every registered condition required')
    conditions={};count=None
    for condition in CONDITIONS:
        p=panels[condition]
        if set(p)!={'recipient','donor'}:raise ValueError('both attribute targets required')
        values={k:{r['benchmark_id']:r['score'] for r in v['items']} for k,v in p.items()}
        ids=set(values['recipient'])
        if not ids or ids!=set(values['donor']) or any(len(values[k])!=len(p[k]['items']) for k in p):
            raise ValueError('attribute score pairing differs')
        if ids!=set(range(len(ids))):raise ValueError('complete registered input indices required')
        count=len(ids) if count is None else count
        if len(ids)!=count:raise ValueError('condition sizes differ')
        if capability=='counting':
            outcomes=[attribute_outcome(values['recipient'][i],values['donor'][i]) for i in sorted(ids)]
            conditions[condition]={k:100*mean(r[k] for r in outcomes)
                for k in ('recipient_count_hit','donor_count_hit','object_present')}
        elif capability=='spatial':
            conditions[condition]={f'{k}_relation_score':100*mean(v.values()) for k,v in values.items()}
        else:raise ValueError('unknown capability')
    expected={'semantic_donor','nuisance_donor','random_sign'}
    if set(norm_flags)!=expected or any(len(v)!=count or any(type(x) is not bool for x in v) for v in norm_flags.values()):
        raise ValueError('norm flags must retain every unit')
    norms={k:dict(matched_units=sum(v),total_units=count,all_matched=all(v)) for k,v in norm_flags.items()}
    contrasts={f'semantic_minus_{c}':{m:conditions['semantic_donor'][m]-conditions[c][m] for m in conditions[c]}
               for c in ('zero','full','nuisance_donor','random_sign','conditional_only')}
    return dict(complete=True,capability=capability,source_seed=source_seed,rows=count,
                conditions=conditions,contrasts=contrasts,norms=norms,
                units='exact hit/presence percent' if capability=='counting' else 'official relation score times 100')


def aggregate_jobs(jobs):
    expected={(c,s) for c in ('counting','spatial') for s in SEEDS}
    if len(jobs)!=8 or {(r['capability'],r['source_seed']) for r in jobs}!=expected:
        raise ValueError('all eight registered jobs required')
    if any(r.get('complete') is not True or r['rows']!=128 or set(r['conditions'])!=set(CONDITIONS) for r in jobs):
        raise ValueError('complete fixed panels required')
    result={}
    for cap in ('counting','spatial'):
        bank=[r for r in jobs if r['capability']==cap]
        if any(r['input_pairs']!=bank[0]['input_pairs'] for r in bank):raise ValueError('cross-run input pairs differ')
        result[cap]={group:{condition:{metric:mean(r[group][condition][metric] for r in bank)
            for metric in bank[0][group][condition]} for condition in bank[0][group]}
            for group in ('conditions','contrasts')}
        result[cap]['all_donor_norms_matched']=all(v['all_matched'] for r in bank for v in r['norms'].values())
        result[cap]['training_runs']=4
    return result
