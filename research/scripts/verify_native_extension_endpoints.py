#!/usr/bin/env python3
"""Reconstruct native reference endpoints from saved answer and official score items."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


SEEDS = (17, 29, 43, 71)
MODELS = ('dimoo', 'januspro', 'showo2')
SIZE = {'dimoo': 512, 'januspro': 384, 'showo2': 432}


def verify_run(report: dict, model: str, capability: str, seed: int) -> dict:
    if model not in MODELS or capability not in ('counting', 'spatial') or seed not in SEEDS:
        raise ValueError('unknown native endpoint identity')
    identity = dict(complete=True, capability=capability, seed=seed)
    if model != 'dimoo':
        identity['backbone'] = model
    if any(report.get(key) != value for key, value in identity.items()):
        raise ValueError('native endpoint identity differs')
    expected_answers = 1024 if capability == 'counting' else 512
    understanding = report['understanding']['128']
    if model == 'dimoo':
        understanding = understanding['raw']
    if understanding['rows'] != expected_answers:
        raise ValueError('native answer denominator differs')
    answer = {}
    for state in ('zero', 'full'):
        result = understanding[state]
        value = result['legacy_accuracy']
        if (result['rows'] != expected_answers or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('native answer accuracy invalid')
        answer[state] = 100 * value
    if abs(answer['full'] - answer['zero'] - understanding['legacy_delta_pp']) > 1e-8:
        raise ValueError('native answer contrast differs')
    generation = {}
    categories = {}
    for state in ('zero', 'full'):
        panel = report['generation'][state]
        items = panel['items']
        if (panel['rows'] != 1000 or len(items) != 1000
                or panel['resolution'] != SIZE[model]
                or panel.get('official_scores_validated') is not True
                or panel.get('staged_images_match') is not True
                or (capability == 'counting' and
                    {item['benchmark_id'] for item in items} != set(range(1000)))
                or (capability == 'spatial' and
                    {(item['category'], item['benchmark_id']) for item in items}
                    != {(category, index) for category in ('2d_spatial', '3d_spatial')
                        for index in range(500)})):
            raise ValueError('incomplete native official image panel')
        scores = [item['score'] for item in items]
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1
               for x in scores):
            raise ValueError('invalid official item scores')
        score = 100 * mean(scores)
        if abs(score - panel['score_x100']) > 1e-8:
            raise ValueError('official item scores disagree with reported mean')
        generation[state] = score
        if capability == 'spatial':
            categories[state] = {}
            for name in ('2d_spatial', '3d_spatial'):
                selected = [item['score'] for item in items if item['category'] == name]
                stated = panel['categories'][name]
                if (len(selected) != 500 or stated['rows'] != 500
                        or abs(100 * mean(selected) - stated['score_x100']) > 1e-8):
                    raise ValueError('spatial official item scores differ')
                categories[state][name] = stated['score_x100']
    if abs(generation['full'] - generation['zero'] - report['generation_delta_x100']) > 1e-8:
        raise ValueError('native generation contrast differs')
    return dict(understanding=answer, generation=generation, categories=categories)


def verify_family(root: Path, model: str) -> dict:
    if model not in MODELS:
        raise ValueError('unknown native model')
    output = {}
    for capability in ('counting', 'spatial'):
        runs = []
        for seed in SEEDS:
            path = root / model / capability / f'seed-{seed}' / 'complete.json'
            if not path.is_file():
                raise FileNotFoundError(f'missing complete native endpoint: {path}')
            runs.append(verify_run(json.loads(path.read_text()), model, capability, seed))
        output[capability] = {
            'understanding': {state: mean(v['understanding'][state] for v in runs)
                              for state in ('zero', 'full')},
            'generation': {state: mean(v['generation'][state] for v in runs)
                           for state in ('zero', 'full')},
            'categories': {state: {name: mean(v['categories'][state][name] for v in runs)
                                    for name in ('2d_spatial', '3d_spatial')}
                           for state in ('zero', 'full')} if capability == 'spatial' else {},
            'runs': runs,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_family(args.root, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
