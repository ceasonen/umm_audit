"""Input-selected decoder diagnostics; comparisons never select attractive outputs."""
from contextlib import contextmanager


@contextmanager
def bypass_layers(layers):
    saved=[(layer,layer.__dict__.get('forward'), 'forward' in layer.__dict__) for layer in layers]
    try:
        for layer in layers:layer.forward=layer.base.forward
        yield
    finally:
        for layer,value,existed in saved:
            if existed:layer.forward=value
            else:del layer.forward


def same_pixels(a,b):
    return a.mode==b.mode and a.size==b.size and a.tobytes()==b.tobytes()


def design(pairs,*,capability,seed):
    if len(pairs)<8 or capability not in ('counting','spatial') or seed!=17:
        raise ValueError('first eight native pairs, both capabilities, reference run 17')
    return dict(capability=capability,seed=seed,uses_scores_for_selection=False,
        units=[dict(index=i,pair=p,decoder_seed=8087+i) for i,p in enumerate(pairs[:8])],
        conditions=['recipient_zero','recipient_full','semantic_donor','direct_donor_zero','bypass_recipient'],
        selection='First eight original input-ordered pairs, including the published failed truck example',
        decoder='All arms reset RNG at the conditioning interface, as in the donor protocol',
        checks=['native captured embeddings equal recorded tensors','zero core equals bypassed projection',
                'native recipient pixels equal historical zero and full','semantic replay equals historical output'],
        scope='Implementation and basic decoder diagnostic; direct donor prompts change text and cannot establish transfer')
