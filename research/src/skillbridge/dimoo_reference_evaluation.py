"""Paired endpoint protocol for the new DiMOO reference-core cohort."""
from skillbridge.dimoo_understanding_contract import inspect_image_tokens, answer_span
from skillbridge.source_measurement import audit_pair, inspect_answer
from skillbridge.evaluation import answers_match


def campaign_plan():
    return dict(jobs=[dict(capability=c,seed=s) for s in (17,29,43,71) for c in ('counting','spatial')],
        states=['zero','full'],answer_budgets=[32,128],understanding_steps='equal to answer budget',
        understanding_answers=24576,generation_images=16000,gpu=1,inference_seed=8087,
        generation_kwargs=dict(height=512,width=512,steps=64,cfg_scale=4.,temperature=1.),
        understanding=dict(system='official',image_size=256,crop_seed_base=20260924,
            temperature=0.,cfg=0.,blocks=1,image_framing='single complete wrapper',
            answer_scoring=['raw','bounded'],include_unterminated_or_masked=True),
        scope='full within-backbone frozen/full endpoints; native objective, model widths and precision differ',
        include_negative_and_zero_effects=True)


def validate_metadata(md,capability,seed):
    expected=dict(backbone='dimoo',method='tied_core',capability=capability,seed=seed,rank=64,
        timestep_buckets=1,image_size=256,gain_limit=.2 if capability=='counting' else .1,
        target_suffixes='q_proj,v_proj')
    if capability not in ('counting','spatial') or seed not in (17,29,43,71) or any(md.get(k)!=v for k,v in expected.items()):
        raise ValueError('DiMOO reference checkpoint identity differs')


def build_query(prompt_tokens,image_tokens,budget):
    if budget not in (32,128) or not prompt_tokens:raise ValueError('unregistered answer budget or missing prompt')
    inspect_image_tokens(image_tokens)
    prefix=list(prompt_tokens[:-1])+list(image_tokens)+list(prompt_tokens[-1:])+[126354]
    if 126336 in prefix:raise ValueError('prompt contains answer mask')
    return prefix+[126336]*budget+[126355],len(prefix)


def response_record(record,*,state,budget,tokens,decode):
    if state not in ('zero','full') or budget not in (32,128):raise ValueError('unregistered condition')
    span=answer_span(tokens,generation_length=budget)
    raw=decode(span['raw_tokens']);bounded=decode(span['answer_tokens'])
    cap=record['capability'];ref=record['answer_text']
    return dict(split_key=record['split_key'],capability=cap,reference=ref,
        question=record['user_prompt'],image=record['user_image'],condition=state,budget=budget,steps=budget,
        prediction=raw,bounded_answer=bounded,token_record=span,
        raw_correct=answers_match(cap,ref,raw),bounded_correct=answers_match(cap,ref,bounded),
        raw_lexical=inspect_answer(cap,raw),bounded_lexical=inspect_answer(cap,bounded))


def validate_response(record,row,*,state,budget,decode):
    expected=response_record(record,state=state,budget=budget,tokens=row['token_record']['raw_tokens'],decode=decode)
    if row!=expected:raise ValueError('saved response, identity, boundary, or scoring differs')


def paired_answers(records,zero,full,*,budget,decode):
    if len(records)!=len(zero) or len(records)!=len(full) or len({r['split_key'] for r in records})!=len(records):
        raise ValueError('paired answer coverage differs')
    for r,z,f in zip(records,zero,full):
        validate_response(r,z,state='zero',budget=budget,decode=decode)
        validate_response(r,f,state='full',budget=budget,decode=decode)
    return dict(raw=audit_pair(zero,full),bounded=audit_pair(
        [{**r,'prediction':r['bounded_answer']} for r in zero],
        [{**r,'prediction':r['bounded_answer']} for r in full]),
        termination={state:dict(rows=len(rows),no_stop=sum(r['token_record']['stop_id'] is None for r in rows),
            unresolved_answer_masks=sum(r['token_record']['unresolved_answer_masks']>0 for r in rows))
            for state,rows in [('zero',zero),('full',full)]})
