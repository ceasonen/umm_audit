from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass

_COUNT_WORDS = {
    "no": 0,
    "none": 0,
    "zero": 0,
    "single": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_COUNT_TOKEN = re.compile(
    r"\b(?:" + "|".join(_COUNT_WORDS) + r"|\d+)\b",
    flags=re.IGNORECASE,
)
_BINARY_POSITIVE = frozenset({"yes", "true"})
_BINARY_NEGATIVE = frozenset({"no", "false"})


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    example_id: str
    capability: str
    reference: str
    prediction: str
    detector_succeeded: bool = True


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _extract_count(value: str) -> int | None:
    for match in _COUNT_TOKEN.finditer(value):
        token = match.group(0).lower()
        if token.isdigit():
            suffix = value[match.end() :]
            if re.match(r"\s*[.)]", suffix):
                continue
            return int(token)
        return _COUNT_WORDS[token]
    return None


def _extract_binary_label(value: str) -> bool | None:
    tokens = normalize_answer(value).split()
    labels: list[bool] = []
    for index, token in enumerate(tokens):
        if token not in _BINARY_POSITIVE | _BINARY_NEGATIVE:
            continue
        label = token in _BINARY_POSITIVE
        if index > 0 and tokens[index - 1] == "not":
            label = not label
        labels.append(label)
    if not labels or any(label != labels[0] for label in labels[1:]):
        return None
    return labels[0]


def answers_match(capability: str, reference: str, prediction: str) -> bool:
    """Compare benchmark answers without treating explanatory prose as an error."""
    if capability == "counting":
        expected = _extract_count(reference)
        observed = _extract_count(prediction)
        if expected is not None and observed is not None:
            return expected == observed
    if capability == "spatial":
        expected_binary = _extract_binary_label(reference)
        observed_binary = _extract_binary_label(prediction)
        if expected_binary is not None:
            return observed_binary is not None and expected_binary == observed_binary
    return normalize_answer(reference) == normalize_answer(prediction)


def response_is_valid(capability: str, prediction: str) -> bool:
    """Return whether a response contains a parseable answer for the capability."""
    if capability == "counting":
        return _extract_count(prediction) is not None
    if capability == "spatial":
        return _extract_binary_label(prediction) is not None
    return bool(normalize_answer(prediction))


def normalized_edit_similarity(reference: str, prediction: str) -> float:
    left = normalize_answer(reference)
    right = normalize_answer(prediction)
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[-1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right), 1)


def evaluate_predictions(records: list[PredictionRecord]) -> dict[str, float]:
    if not records:
        msg = "prediction records must be non-empty"
        raise ValueError(msg)
    exact = [
        answers_match(item.capability, item.reference, item.prediction) for item in records
    ]
    valid = [response_is_valid(item.capability, item.prediction) for item in records]
    edit = [normalized_edit_similarity(item.reference, item.prediction) for item in records]
    metrics: dict[str, float] = {
        "count": float(len(records)),
        "exact_match": sum(exact) / len(records),
        "normalized_edit_similarity": sum(edit) / len(records),
        "detector_success": sum(item.detector_succeeded for item in records) / len(records),
        "valid_response": sum(valid) / len(records),
    }
    by_capability: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        by_capability[item.capability].append(index)
    for capability, indices in sorted(by_capability.items()):
        metrics[f"{capability}/exact_match"] = sum(exact[index] for index in indices) / len(indices)
        metrics[f"{capability}/valid_response"] = sum(
            valid[index] for index in indices
        ) / len(indices)
        if capability == "spatial":
            invalid = sum(
                _extract_binary_label(records[index].prediction) is None for index in indices
            )
            metrics["spatial/invalid_response"] = invalid / len(indices)
    return metrics
