"""Image-dependence diagnostics; not isolated semantic counterfactuals."""
import random
from pathlib import Path

from PIL import Image

from skillbridge.full_answer_audit import validate_resume

SEEDS = (17, 29, 43, 71)


def image_assignment(records):
    """A label-blind image permutation fixed across all training seeds."""
    n = len(records)
    if n < 2 or len({r['split_key'] for r in records}) != n or len({r['user_image'] for r in records}) != n:
        raise ValueError('at least two unique record identities and images required')
    rng = random.Random(20260916)
    assignment = list(range(n))
    for _ in range(10000):
        rng.shuffle(assignment)
        if all(i != j for i, j in enumerate(assignment)):
            return assignment
    raise ValueError('could not construct image derangement')


def make_cases(seed):
    if seed not in SEEDS:
        raise ValueError('unsupported training seed')
    cases = [dict(state=s, verbalizer=v, image_control=c)
             for s in ('full', 'zero', 'label_shuffle') for v in ('digits', 'words')
             for c in ('neutral', 'permuted')]
    random.Random(20260916 + seed).shuffle(cases)
    return cases


def controlled_records(records, control, assignment, output):
    if assignment != image_assignment(records) or control not in ('neutral', 'permuted'):
        raise ValueError('image-control design differs from fixed assignment')
    result = []
    output = Path(output)
    for i, row in enumerate(records):
        if control == 'permuted':
            path = records[assignment[i]]['user_image']
        else:
            with Image.open(row['user_image']) as original:
                size = original.size
            output.mkdir(parents=True, exist_ok=True)
            path = output/f'neutral-{size[0]}x{size[1]}.png'
            if not path.exists():
                Image.new('RGB', size, (127, 127, 127)).save(path)
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != size or image.getextrema() != ((127, 127),)*3:
                    raise ValueError('saved neutral image changed')
        result.append({**row, 'user_image': str(path)})
    return result


def validate_predictions(original, controlled, assignment, predictions, case):
    count = validate_resume(controlled, predictions, 'counting', case['state'], case['verbalizer'])
    for i, saved in enumerate(predictions):
        expected = dict(image_control=case['image_control'], original_image=original[i]['user_image'],
                        donor_split_key=original[assignment[i]]['split_key'] if case['image_control']=='permuted' else None)
        if any(saved.get(k) != v for k, v in expected.items()) or saved['max_new_tokens'] != 32:
            raise ValueError('image-control provenance or decoding limit differs')
    return count


def paired_effects(original, control):
    ids = [r['split_key'] for r in original['zero']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate paired identity')
    for panels in (original, control):
        for state in ('full', 'zero', 'label_shuffle'):
            if [r['split_key'] for r in panels[state]] != ids:
                raise ValueError('paired identity or order changed')
            if any(type(r['semantic_correct']) is not bool for r in panels[state]):
                raise ValueError('explicit per-answer correctness required')
    def mean(panel, state):
        return 100*sum(r['semantic_correct'] for r in panel[state])/len(ids)
    def delta(panel, baseline):
        return mean(panel, 'full') - mean(panel, baseline)
    return dict(rows=len(ids), full_minus_zero_original_pp=delta(original, 'zero'),
                full_minus_zero_control_pp=delta(control, 'zero'),
                image_dependent_gain_pp=delta(original, 'zero')-delta(control, 'zero'),
                full_minus_shuffle_image_dependent_gain_pp=delta(original, 'label_shuffle')-delta(control, 'label_shuffle'),
                scope='image-dependence diagnostic; not attribute-specific mediation or independent confirmation')
