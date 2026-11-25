"""Protocol checks for the new, one-bucket DiMOO reference cohort."""
from contextlib import contextmanager
import random

from skillbridge.dimoo_understanding_contract import inspect_image_tokens


@contextmanager
def preprocessing_rng(seed):
    """Fix native crop/mask randomness without changing caller RNG streams."""
    import numpy as np
    python_state=random.getstate(); numpy_state=np.random.get_state()
    try:
        random.seed(seed); np.random.seed(seed)
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def validate_reference(config,capability,seed):
    if capability not in ('counting','spatial') or seed not in (17,29,43,71):
        raise ValueError('unregistered capability or training run')
    steps=8192 if capability=='counting' else 4096
    expected=dict(backbone='blip3o',capability=capability,seed=seed,method='tied_core',
        rank=64,timestep_buckets=1,steps=steps,examples=steps,optimizer='adamw',
        learning_rate=.0001,weight_decay=.01,gain_limit=.2 if capability=='counting' else .1,
        target_suffixes=['q_proj','v_proj'],preserve_source_order=True,shuffle_source_labels=False)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('reference training recipe differs')


def audit_native_batch(inputs,labels,*,prompt_tokens,answer_tokens):
    """Check reconstructed native targets against actual tokenized annotations."""
    x=list(inputs);y=list(labels)
    if len(x)!=len(y) or x.count(126354)!=1 or x.count(126349)!=1 or x.count(126350)!=1:
        raise ValueError('batch shape or protocol boundaries differ')
    start=x.index(126354)+1; left=x.index(126349);right=x.index(126350)+1
    grid=inspect_image_tokens(x[left:right])
    if x[:left]+x[right:start-1]!=list(prompt_tokens):
        raise ValueError('training prompt differs')
    if any(v!=-100 for v in y[:start]):raise ValueError('instruction or image is supervised')
    if len(x)-start!=1024:raise ValueError('native answer and padding length differs')
    if not answer_tokens or answer_tokens[-1]!=126355 or len(answer_tokens)>1024:
        raise ValueError('answer annotation lacks native ending')
    if any(a!=126336 for a,b in zip(x,y) if b!=-100):
        raise ValueError('supervised target was not masked')
    reconstructed=[b if b!=-100 else a for a,b in zip(x[start:],y[start:])]
    expected=list(answer_tokens)+[126339]*(1024-len(answer_tokens))
    if reconstructed!=expected:raise ValueError('training targets differ from reference answer')
    supervised=[b for b in y[start:] if b!=-100]
    answer=sum(b not in (126339,126355) for b in supervised)
    if answer<1:raise ValueError('no answer-content supervision')
    return dict(sequence_length=len(x),answer_span_length=1024,supervised_tokens=len(supervised),
        answer_tokens=answer,padding_tokens=supervised.count(126339),end_tokens=supervised.count(126355),
        grid=grid)


def validate_contract(report):
    from skillbridge.dimoo_understanding_contract import diagnostic_cases
    expected={(c['name'],i,s) for c in diagnostic_cases() for i in (0,1,128,129) for s in ('zero','full')}
    rows=report.get('reports',[])
    actual=[(r.get('case'),r.get('index'),r.get('state')) for r in rows]
    if (report.get('complete') is not True or report.get('answers')!=40 or
            report.get('historical_replays')!=8 or len(actual)!=40 or set(actual)!=expected):
        raise ValueError('complete fixed protocol diagnostic required')
    if any(r.get('historical_replay_equal') is not True for r in rows if r['case']=='historical'):
        raise ValueError('historical inference contract did not replay')
