"""Locked, same-checkpoint source diagnostics for the original paper cohort."""
from pathlib import Path
import random

from skillbridge.evaluation import answers_match
from skillbridge.source_measurement import inspect_answer


def validate_checkpoint(path, metadata, capability, seed):
    expected = ('runs','oral_scale_calibrated_v2','tied_core','blip3o',capability,f'seed-{seed}','core.pt')
    if tuple(Path(path).parts[-7:]) != expected:
        raise ValueError('checkpoint must belong to the original core cohort')
    required = dict(backbone='blip3o', capability=capability, seed=seed, method='tied_core',
                    rank=64, timestep_buckets=1, gain_limit=.2 if capability=='counting' else .1,
                    target_suffixes='q_proj,v_proj')
    if any(metadata.get(k)!=v for k,v in required.items()):
        raise ValueError('original checkpoint identity or parameterization differs')


def cases(seed):
    if seed not in (17,29,43,71):
        raise ValueError('unregistered original training run')
    result=[]
    # Complete original-image instruction controls first; no selection by outcomes.
    for control in ('original','permuted','neutral'):
        block=[dict(state=s,image_control=control,instruction=i)
               for i in ('original','answer_only','answer_only_rephrased') for s in ('zero','full')]
        random.Random(20260923+seed+len(result)).shuffle(block)
        result.extend(block)
    return result


def score_answer(prediction, reference, capability):
    observed=inspect_answer(capability,prediction)
    target=inspect_answer(capability,reference)
    if target['review_required']:
        raise ValueError('reference is not unambiguous')
    return dict(legacy_correct=answers_match(capability,reference,prediction),
                unambiguous_correct=not observed['review_required'] and observed['candidate']==target['candidate'],
                lexical_prediction=observed['candidate'],answer_only=observed['answer_only'],
                requires_review=observed['review_required'],flags=observed['flags'])


def summarize(rows,pairs):
    if not rows or len({r['split_key'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate prediction rows')
    bank={r['split_key']:r for r in rows}
    if any(a not in bank or b not in bank for a,b in pairs):
        raise ValueError('natural pair absent from predictions')
    both=[bank[a]['unambiguous_correct'] and bank[b]['unambiguous_correct'] for a,b in pairs]
    return dict(rows=len(rows),legacy_accuracy=sum(r['legacy_correct'] for r in rows)/len(rows),
                unambiguous_correct_rate=sum(r['unambiguous_correct'] for r in rows)/len(rows),
                answer_only_rate=sum(r['answer_only'] for r in rows)/len(rows),
                review_rate=sum(r['requires_review'] for r in rows)/len(rows),
                natural_pairs=len(pairs),natural_pair_both_unambiguous_correct=sum(both)/len(both) if both else None,
                interpretation='Lexical diagnostics, not human adjudication; ambiguous responses remain in denominator')


def validate_resume(records,rows,case,seed):
    if len(rows)>len(records):
        raise ValueError('too many saved answers')
    for r,p in zip(records,rows):
        expected=dict(split_key=r['split_key'],image=r['user_image'],question=r['user_prompt'],
                      reference=r['answer_text'],source_seed=seed,**case)
        if any(p.get(k)!=v for k,v in expected.items()):
            raise ValueError('saved answer differs from checkpoint, input, or condition')
    return len(rows)


def verify_idle_context_placement(argv, gpu):
    """Check explicit idle PIDs against the two observed foreign simulator layouts.

    This only verifies command placement. Admission still requires an explicit
    PID list, <=256 MiB per context, low utilization, and enough free memory.
    """
    device_flags = [i for i, value in enumerate(argv)
                    if value in ('--device', '--gpu-index')]
    if len(device_flags) != 1 or device_flags[0] + 1 >= len(argv):
        raise ValueError('foreign context requires one unambiguous device argument')
    index = device_flags[0]
    option, value = argv[index:index+2]
    if option == '--device' and value == 'cuda:0' and gpu in (1, 2, 3):
        return True
    if (option == '--gpu-index' and value in ('4', '5', '6', '7')
            and gpu in (0, 1, 2, 3) and len(argv) > 1
            and argv[1] == 'tools/v3/worldgen/isaac_structure_latent_pilot.py'
            and argv.count('--repo') == 1):
        repo_index = argv.index('--repo')
        if argv[repo_index+1:repo_index+2] == ['/mnt/sdc/zz/mtare_topo_comm']:
            return True
    raise ValueError('foreign context placement does not match a verified simulator')


def admit_gpu(gpu,free_mib,utilization,processes,*,allowed_idle_pids):
    if gpu not in range(4) or free_mib<20000 or utilization>5:
        raise ValueError('GPU outside project allowance or insufficient idle capacity')
    if any(p['pid'] not in allowed_idle_pids or not 0<=p['memory_mib']<=256 for p in processes):
        raise ValueError('GPU has an unapproved process or substantial allocation')
    return True


def full_replay(rows,historical):
    if len(rows)!=len(historical) or any(a['split_key']!=b['split_key'] for a,b in zip(rows,historical)):
        raise ValueError('full historical replay input coverage differs')
    different=[a['split_key'] for a,b in zip(rows,historical) if a['prediction']!=b['prediction']]
    return dict(rows=len(rows),mismatches=len(different),mismatch_ids=different,passed=not different)


def contrasts(cells):
    keys=('image_control','instruction','state')
    expected={tuple(c[k] for k in keys) for c in cases(17)}
    bank={tuple(c[k] for k in keys):c for c in cells}
    if len(cells)!=18 or set(bank)!=expected:raise ValueError('all 18 fixed cells required')
    metrics=['legacy_accuracy','unambiguous_correct_rate','answer_only_rate','review_rate','natural_pair_both_unambiguous_correct']
    result={}
    for control in ('original','permuted','neutral'):
        for instruction in ('original','answer_only','answer_only_rephrased'):
            a,b=(bank[control,instruction,s] for s in ('zero','full'))
            result[control+'/'+instruction]={k+'_delta_pp':100*(b[k]-a[k]) if b[k] is not None and a[k] is not None else None for k in metrics}
    return result
