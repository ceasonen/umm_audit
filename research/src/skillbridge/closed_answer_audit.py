"""Fixed-answer measurement and natural image contrasts, without output selection."""
from collections import defaultdict
import math


def validate_design(design, complete, *, seed, backbone, capability):
    identity = (seed, backbone, capability)
    for value in (design, complete):
        if tuple(value[k] for k in ('seed', 'backbone', 'capability')) != identity:
            raise ValueError('source panel identity mismatch')
    forms = ('digits', 'words') if capability == 'counting' else ('yes_no', 'truth')
    expected = {(s, v) for s in ('zero', 'full', 'label_shuffle') for v in forms}
    actual = [(case['state'], case['verbalizer']) for case in design['cases']]
    if len(actual) != 6 or set(actual) != expected or complete['cases'] != 6:
        raise ValueError('exactly six unique source controls/verbalizers required')
    records = design['records']
    keys = [r['split_key'] for r in records]
    if not keys or len(keys) != len(set(keys)) or complete['rows_per_case'] != len(keys):
        raise ValueError('empty, duplicate or incomplete source panel')


def validate_image_disjointness(train, test):
    from skillbridge.unseen_source_panel import image_identity

    overlap = ({image_identity(r['user_image']) for r in train}
               & {image_identity(r['user_image']) for r in test})
    if overlap:
        raise ValueError(f'source image identities overlap: {len(overlap)}')
from pathlib import Path
import random
import json


def answer_design(capability, verbalizer):
    designs = {
        ('counting','digits'): (['1','2','3','4'], ['1','2','3','4'], '\nAnswer with a single digit.'),
        ('counting','words'): (['1','2','3','4'], ['one','two','three','four'], '\nAnswer with one lowercase number word.'),
        ('spatial','yes_no'): (['no','yes'], ['no','yes'], '\nAnswer with one lowercase word: yes or no.'),
        ('spatial','truth'): (['no','yes'], ['false','true'], '\nAnswer with one lowercase word: true or false.'),
    }
    labels, answers, suffix = designs[capability, verbalizer]
    return dict(labels=labels, answers=answers, suffix=suffix)


def single_token_ids(encoded):
    if not encoded or any(len(x)!=1 for x in encoded):
        raise ValueError('fixed answers must each occupy exactly one continuation token')
    ids=[x[0] for x in encoded]
    if len(set(ids))!=len(ids):
        raise ValueError('candidate token IDs must be distinct')
    return ids


def classify(log_probs, labels, reference):
    if (len(labels)<2 or len(labels)!=len(log_probs) or len(set(labels))!=len(labels)
            or reference not in labels or any(not math.isfinite(x) or x>0 for x in log_probs)):
        raise ValueError('aligned finite log probabilities and an in-set reference required')
    top=max(log_probs)
    winners=[label for label,p in zip(labels,log_probs) if p==top]
    prediction=winners[0] if len(winners)==1 else None
    score=log_probs[labels.index(reference)]
    return dict(prediction=prediction, correct=prediction==reference, tie=len(winners)>1,
        margin=score-max(p for p,label in zip(log_probs,labels) if label!=reference),
        candidate_log_mass=top+math.log(sum(math.exp(p-top) for p in log_probs)))


def natural_pairs(records):
    if len({r['split_key'] for r in records})!=len(records):
        raise ValueError('duplicate record IDs')
    groups, annotations = defaultdict(list), {}
    for r in records:
        key=(r['user_prompt'].strip(),Path(r['user_image']).name)
        if key in annotations and annotations[key]!=r['answer_text']:
            raise ValueError('conflicting annotation for the same image/question')
        annotations[key]=r['answer_text']
        groups[key[0]].append(r)
    rng=random.Random(20260913)
    questions=sorted(groups)
    rng.shuffle(questions)
    used, pairs=set(),[]
    for question in questions:
        rows=sorted(groups[question],key=lambda r:r['split_key'])
        rng.shuffle(rows)
        # At most one disjoint pair per question; never select using model scores.
        pair=next(((a,b) for i,a in enumerate(rows) for b in rows[i+1:]
                   if a['answer_text']!=b['answer_text']
                   and Path(a['user_image']).name!=Path(b['user_image']).name
                   and all(Path(r['user_image']).name not in used for r in (a,b))),None)
        if pair:
            pairs.append([r['split_key'] for r in pair])
            used.update(Path(r['user_image']).name for r in pair)
    return pairs


def contrast_outcomes(records, predictions, pairs):
    rows={r['split_key']:r for r in records}
    scored={r['split_key']:r for r in predictions}
    if len(scored)!=len(predictions) or set(scored)!=set(rows):
        raise ValueError('prediction coverage differs from locked records')
    output=[]
    for a,b in pairs:
        ra,rb,pa,pb=rows[a],rows[b],scored[a],scored[b]
        if (ra['user_prompt'].strip()!=rb['user_prompt'].strip()
                or ra['answer_text']==rb['answer_text'] or pa['labels']!=pb['labels']
                or pa['reference']!=ra['answer_text'] or pb['reference']!=rb['answer_text']):
            raise ValueError('invalid natural contrast or scored reference')
        ia,ib=(pa['labels'].index(r['answer_text']) for r in (ra,rb))
        shift=(pa['log_probs'][ia]-pa['log_probs'][ib])-(pb['log_probs'][ia]-pb['log_probs'][ib])
        output.append(dict(pair=[a,b],correct_log_odds_shift=shift,
            both_correct=pa['prediction']==ra['answer_text'] and pb['prediction']==rb['answer_text']))
    return output


def resume_count(records,predictions,*,state,verbalizer):
    if len(predictions)>len(records):
        raise ValueError('too many saved predictions')
    for r,p in zip(records,predictions):
        fields={'split_key':r['split_key'],'image':r['user_image'],'question':r['user_prompt'],
                'reference':r['answer_text'],'state':state,'verbalizer':verbalizer,
                'labels':answer_design(r['capability'],verbalizer)['labels']}
        if any(p[k]!=v for k,v in fields.items()):
            raise ValueError('saved prediction does not match locked condition and inputs')
    return len(predictions)


def validate_training(directory,*,shuffled):
    from skillbridge.training_overlap import _validate_artifacts
    config=json.loads((directory/'training-config.json').read_text())
    if config['shuffle_source_labels'] is not shuffled:
        raise ValueError('wrong normal/shuffled training condition')
    _validate_artifacts(directory,config,['adapter.pt','core.pt','history.jsonl','summary.json',
        'training_order.json','training-config.json','training-inputs.jsonl'])
    return config
