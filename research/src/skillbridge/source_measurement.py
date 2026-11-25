"""Non-destructive answer-format diagnostics, separate from benchmark scoring."""
from __future__ import annotations

import re
import random
from collections import Counter
from pathlib import Path

WORDS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(21), strict=True))
WORDS.update(no=0, none=0, single=1)
NUMBER = re.compile(r'\b(?:' + '|'.join(WORDS) + r'|\d+)\b', re.I)
TEMPLATES = {
    'answer_only': {
        'counting': '\nAnswer with a single non-negative integer only.',
        'spatial': '\nAnswer with exactly one word: true or false.',
    },
    'answer_only_rephrased': {
        'counting': '\nReturn only the number of objects asked about, as digits. Do not explain.',
        'spatial': '\nReturn true if the statement matches the image, otherwise false. Do not explain.',
    },
}


def inspect_answer(capability: str, prediction: str) -> dict:
    """Flag lexical uncertainty; this is a heuristic audit, not human adjudication."""
    if not isinstance(prediction, str) or capability not in ('counting', 'spatial'):
        raise ValueError('string prediction and counting/spatial capability required')
    text = prediction.strip().lower()
    flags = []
    if re.search(r'\b(maybe|perhaps|probably|likely|uncertain|ambiguous|could|might)\b|'
                 r'not sure|cannot (tell|determine)|can.t (tell|determine)', text):
        flags.append('uncertainty')
    if re.search(r'\bnot\b|\bno (fewer|more|less|greater)\b', text):
        flags.append('negation')
    if capability == 'counting':
        tokens = list(NUMBER.finditer(text))
        values = [int(m[0]) if m[0].isdigit() else WORDS[m[0]] for m in tokens]
        if re.search(r'\d[./]\d|(?<!\w)[-−+]\d|\d\s*[-–]\s*\d', text):
            flags.append('non_integer_notation')
        for a, b in zip(tokens, tokens[1:]):
            if a[0] in WORDS and b[0] in WORDS and text[a.end():b.start()].strip(' -') == '':
                flags.append('compound_number')
                break
        answer_only = bool(re.fullmatch(r'(?:' + '|'.join(WORDS) + r'|\d+)[.!]?', text))
        if re.search(r'\d[.)](?:\s|$)', text):
            flags.append('numeric_terminal_punctuation')
    else:
        tokens = re.findall(r'\b(yes|true|no|false)\b', text)
        values = [t in ('yes', 'true') for t in tokens]
        answer_only = bool(re.fullmatch(r'(yes|true|no|false)[.!]?', text))
    if not values:
        flags.append('no_candidate')
    if len(set(values)) > 1:
        flags.append('multiple_values')
    review = any(f != 'numeric_terminal_punctuation' for f in flags)
    return dict(candidate=None if review else values[0], answer_only=answer_only,
                flags=flags, review_required=review, characters=len(prediction),
                truncation_status='unknown_without_generated_token_metadata')


def audit_rows(rows: list[dict]) -> tuple[dict, list[dict]]:
    from skillbridge.evaluation import answers_match

    seen, detailed = set(), []
    for row in rows:
        key = row['split_key']
        if not key or key in seen:
            raise ValueError('empty or duplicate prediction identity')
        seen.add(key)
        expected = inspect_answer(row['capability'], row['reference'])
        if expected['review_required'] or not expected['answer_only']:
            raise ValueError('reference must be an unambiguous answer')
        audit = inspect_answer(row['capability'], row['prediction'])
        detailed.append({**row, **audit,
            'legacy_correct': answers_match(row['capability'], row['reference'], row['prediction']),
            'audit_correct': None if audit['review_required'] else audit['candidate'] == expected['candidate']})
    if not detailed:
        raise ValueError('empty predictions')
    n = len(detailed)
    return dict(rows=n, legacy_accuracy=sum(r['legacy_correct'] for r in detailed)/n,
        answer_only_rate=sum(r['answer_only'] for r in detailed)/n,
        review_rows=sum(r['review_required'] for r in detailed),
        heuristic_correct_rate=sum(r['audit_correct'] is True for r in detailed)/n,
        legacy_audit_disagreements=sum(r['audit_correct'] is not None and
            r['legacy_correct'] != r['audit_correct'] for r in detailed),
        flags=dict(Counter(f for r in detailed for f in r['flags'])),
        scope='lexical sensitivity audit; ambiguous rows retained, not semantic ground truth'), detailed


def audit_pair(zero: list[dict], full: list[dict]) -> dict:
    a, ar = audit_rows(zero)
    b, br = audit_rows(full)
    fields = ('split_key', 'capability', 'reference', 'question', 'image')
    if len(ar) != len(br) or any(any(x[f] != y[f] for f in fields) for x, y in zip(ar, br)):
        raise ValueError('paired identities, questions, images and references must agree')
    both = [(x,y) for x,y in zip(ar,br) if not x['review_required'] and not y['review_required']]
    return dict(rows=len(ar), zero=a, full=b,
        legacy_delta_pp=100*(b['legacy_accuracy']-a['legacy_accuracy']),
        heuristic_delta_pp=100*(b['heuristic_correct_rate']-a['heuristic_correct_rate']),
        both_unambiguous_rows=len(both),
        both_unambiguous_delta_pp=(100*sum(int(y['audit_correct'])-int(x['audit_correct'])
            for x,y in both)/len(both)) if both else None,
        conditional_subset_warning='both-unambiguous subset depends on model outputs; not a causal estimand')


def template_records(records: list[dict], template: str) -> list[dict]:
    suffixes = TEMPLATES[template]
    return [{**r, 'user_prompt': r['user_prompt'] + suffixes[r['capability']]} for r in records]


def disjoint_records(train: list[dict], test: list[dict]) -> None:
    if {Path(r['user_image']).name for r in train} & {Path(r['user_image']).name for r in test}:
        raise ValueError('source/test image overlap')


def can_measure_alongside(gpu, seed, status, *, free_mib, processes):
    if gpu not in range(4) or seed != (17,29,43,71)[gpu]:
        raise ValueError('only registered seed on physical GPU 0 through 3 permitted')
    return (status.get('status') == 'running' and status.get('gpu') == gpu
        and status.get('seed') == seed and status.get('job', '').endswith('-base-generation')
        and free_mib >= 11500 and processes == 1)


def measurement_cases(*, seed: int) -> list[dict]:
    rng = random.Random(seed + 20260913)
    capabilities = ['counting','spatial']
    rng.shuffle(capabilities)
    cases = []
    for cap in capabilities:
        block = [dict(capability=cap,template=template,state=state)
                 for template in TEMPLATES for state in ('zero','reference','label_shuffle')]
        rng.shuffle(block)
        cases.extend(block)
    return cases


def validate_completed_predictions(records, predictions):
    pairs = [('split_key','split_key'),('capability','capability'),('answer_text','reference'),
             ('user_image','image'),('user_prompt','question')]
    if len(records) != len(predictions) or any(
        any(a[x] != b[y] for x,y in pairs) for a,b in zip(records,predictions)):
        raise ValueError('completed predictions disagree with locked input records')
    if len({r['split_key'] for r in predictions}) != len(predictions):
        raise ValueError('duplicate completed prediction identities')
