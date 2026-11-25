"""Complete-answer measurement with explicitly separate format and meaning.

This conservative recognizer accepts answer-only aliases, not arbitrary prose.
Unrecognized/truncated answers stay in the denominator and require review; its
accuracy must not be advertised as a complete human judgment of free text.
"""
import re

from skillbridge.closed_answer_audit import answer_design

WORDS = ('zero one two three four five six seven eight nine ten eleven twelve '
         'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty').split()
NUMBER_WORDS = {word: str(i) for i, word in enumerate(WORDS)}
RULE_VERSION = 'full-answer-v1-answer-only-aliases'


def classify_answer(text, capability, verbalizer, reference, *, truncated):
    design = answer_design(capability, verbalizer)
    if reference not in design['labels'] or not isinstance(text, str):
        raise ValueError('invalid reference or answer type')
    if not isinstance(truncated, bool):
        raise ValueError('explicit truncation flag required')
    stripped = text.strip()
    normalized = stripped.casefold()
    # One terminal full stop is harmless; internal punctuation is not erased.
    if normalized.endswith('.'):
        normalized = normalized[:-1].strip()
    prediction = None
    compliant = False
    if not truncated:
        if capability == 'counting':
            prediction = NUMBER_WORDS.get(normalized)
            if re.fullmatch(r'[0-9]+', normalized):
                prediction = str(int(normalized))
            compliant = (bool(re.fullmatch(r'[0-9]', stripped)) if verbalizer == 'digits'
                         else stripped in NUMBER_WORDS)
        else:
            prediction = {'yes': 'yes', 'true': 'yes', 'no': 'no', 'false': 'no'}.get(normalized)
            compliant = stripped in design['answers']
    return dict(raw_answer=text, reference=reference, truncated=truncated,
                rule_version=RULE_VERSION, format_compliant=compliant,
                semantic_prediction=prediction, semantic_correct=prediction == reference,
                requires_review=prediction is None)


def generated_span(sequence, score_steps, eos_ids, limit):
    if not 0 < score_steps <= min(len(sequence), limit) or not eos_ids:
        raise ValueError('invalid generated-token boundary')
    tokens = list(sequence[-score_steps:])
    ended = tokens[-1] in eos_ids
    if any(token in eos_ids for token in tokens[:-1]):
        raise ValueError('unexpected output after EOS')
    if not ended and score_steps < limit:
        raise ValueError('unexplained early stop; do not score as a complete answer')
    return dict(generated_token_ids=tokens, generated_steps=score_steps,
                ended_with_eos=ended, truncated=not ended)


def validate_resume(records, predictions, capability, state, verbalizer):
    if len(predictions) > len(records):
        raise ValueError('too many saved predictions')
    for row, saved in zip(records, predictions):
        expected = dict(split_key=row['split_key'], image=row['user_image'],
                        question=row['user_prompt'], reference=row['answer_text'],
                        state=state, verbalizer=verbalizer)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('resume input or condition mismatch')
        span = generated_span(saved['generated_token_ids'], saved['generated_steps'],
                              saved['eos_token_ids'], saved['max_new_tokens'])
        measured = classify_answer(saved['raw_answer'], capability, verbalizer,
                                   row['answer_text'], truncated=span['truncated'])
        if any(saved.get(k) != v for k, v in {**span, **measured}.items()):
            raise ValueError('saved token boundary or classification disagrees')
    return len(predictions)


def summarize(predictions, pairs):
    by_id = {row['split_key']: row for row in predictions}
    if not predictions or len(by_id) != len(predictions):
        raise ValueError('empty or duplicate predictions')
    if any(a not in by_id or b not in by_id for a, b in pairs):
        raise ValueError('missing paired prediction')
    both = [by_id[a]['semantic_correct'] and by_id[b]['semantic_correct'] for a, b in pairs]
    return dict(rows=len(predictions),
                semantic_answer_accuracy=sum(r['semantic_correct'] for r in predictions)/len(predictions),
                format_compliance=sum(r['format_compliant'] for r in predictions)/len(predictions),
                unrecognized_rate=sum(r['requires_review'] for r in predictions)/len(predictions),
                truncated_rate=sum(r['truncated'] for r in predictions)/len(predictions),
                natural_pairs=len(pairs), natural_pair_both_correct=sum(both)/len(both) if both else None,
                rule_version=RULE_VERSION,
                interpretation='answer-only semantic recognition; prose retained for blind review; not proof of understanding')
