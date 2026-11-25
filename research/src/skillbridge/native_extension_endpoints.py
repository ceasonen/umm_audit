"""Registered native endpoint protocols for the two trained extensions."""
import json
from pathlib import Path


def endpoint_plan(backbone):
    if backbone not in ('januspro', 'showo2'):
        raise ValueError('unknown native extension backbone')
    size = 384 if backbone == 'januspro' else 432
    generation = (dict(width=384, height=384, cfg_weight=5.0,
                       temperature=1.0, image_token_num_per_image=576)
                  if backbone == 'januspro' else
                  dict(width=432, height=432, guidance_scale=7.5,
                       num_inference_steps=50))
    return dict(backbone=backbone, image_size=size, generation_kwargs=generation,
                answer_budgets=[32, 128], inference_seed=8087,
                understanding_answers=24576, generation_images=16000,
                jobs=[dict(capability=capability, seed=seed)
                      for seed in (17, 29, 43, 71)
                      for capability in ('counting', 'spatial')],
                states=['zero', 'full'])


def validate_training_configuration(config, backbone, capability, seed):
    if backbone not in ('januspro', 'showo2'):
        raise ValueError('unknown training backbone')
    steps = 8192 if capability == 'counting' else 4096
    expected = dict(backbone=backbone, capability=capability, seed=seed,
                    method='tied_core', steps=steps, examples=steps,
                    rank=64, timestep_buckets=1,
                    image_size=endpoint_plan(backbone)['image_size'],
                    optimizer='adamw', learning_rate=.0001, weight_decay=.01,
                    gain_limit=.2 if capability == 'counting' else .1,
                    preserve_source_order=True, shuffle_source_labels=False)
    if any(config.get(key) != value for key, value in expected.items()) or \
            tuple(config.get('target_suffixes', ())) != ('q_proj', 'v_proj'):
        raise ValueError('completed training configuration differs from registered protocol')


def frozen_reuse_provider(provider: Path, backbone: str, capability: str) -> Path:
    marker=provider/'complete.json'
    if not marker.is_file():raise ValueError('completed frozen provider missing')
    report=json.loads(marker.read_text())
    expected=dict(complete=True,backbone=backbone,capability=capability,seed=17)
    if any(report.get(key)!=value for key,value in expected.items()):
        raise ValueError('frozen provider identity differs')
    return provider


def prepare_native_frozen_reuse(base,out,backbone,capability,seed,understanding,generation,
                                first_answer,first_image):
    """Link a completed frozen panel only after an independent live parity probe."""
    if seed==17:return None
    provider=base.parent/'seed-17/endpoints'
    if not (provider/'complete.json').is_file():return None
    frozen_reuse_provider(provider,backbone,capability)
    design=json.loads((provider/'design.json').read_text())
    if (design.get('protocol')!=endpoint_plan(backbone) or
            design.get('understanding_records')!=understanding or
            design.get('generation_records')!=generation or
            design.get('source_selection_uses_target') is not False):
        raise ValueError('frozen provider input or inference protocol differs')
    if json.loads((provider/'zero-bypass-parity.json').read_text())!=dict(answer_equal=True,pixels_equal=True):
        raise ValueError('frozen provider native parity differs')
    from PIL import Image
    from skillbridge.decoder_sanity import same_pixels
    first=json.loads((provider/'understanding/tokens-32/zero/predictions.jsonl').read_text().splitlines()[0])
    if first.get('prediction')!=first_answer:raise ValueError('live frozen answer differs from provider')
    with Image.open(provider/'generation/zero/images/00000.png') as historical:
        if not same_pixels(first_image,historical):raise ValueError('live frozen image differs from provider')
    relatives=('understanding/tokens-32/zero','understanding/tokens-128/zero','generation/zero')
    for relative in relatives:
        destination=out/relative
        if destination.exists() and not destination.is_symlink():return None
        if destination.is_symlink() and destination.resolve()!=(provider/relative).resolve():
            raise ValueError('existing frozen reuse names another provider')
    for relative in relatives:
        destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_symlink():destination.symlink_to(provider/relative,target_is_directory=True)
    report=dict(provider=str(provider),provider_seed=17,consumer_seed=seed,
                backbone=backbone,capability=capability,reused_answers=2*len(understanding),
                reused_images=len(generation),scope='identical native frozen model, inputs and paired noise')
    from scripts.run_uniworld_replication_endpoints import write
    write(out/'baseline-reuse.json',report)
    return report
