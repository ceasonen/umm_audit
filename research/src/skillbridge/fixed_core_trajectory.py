"""Fixed parameter negation and complete native update-energy trajectory auditing."""
import math
import torch
from skillbridge.sidecar import RoutedLowRankLinear
from skillbridge.readout_followup import native_route_outputs


def negated_state(state):
    if not state or any(v.ndim != 2 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('finite core matrices required')
    return {name: -value.detach().clone() for name,value in state.items()}


class TrajectoryObserver:
    """Observe every routed call; never replace activations or parameters.

    The norm concerns native local updates summed over the actual trajectory.
    It is not distance between entire hidden-state trajectories and does not
    assume token correspondence when generated answer lengths differ.
    """
    def __init__(self, model):
        self.layers={f'{n}.capability_delta' if n else 'capability_delta':layer
                     for n,layer in model.named_modules() if isinstance(layer,RoutedLowRankLinear)}
        if not self.layers:raise ValueError('no core sites')
        self.rows={};self.handles=[]

    def __enter__(self):
        if self.handles:raise ValueError('observer already active')
        self.rows={name:dict(squared_update_norm=0.,elements=0,calls=0,multi_token_calls=0,
                            single_token_calls=0) for name in self.layers}
        for name,layer in self.layers.items():
            def hook(module,args,output,site=name):
                with torch.no_grad():
                    baseline,actual=native_route_outputs(module,args[0],module.capability_delta)
                    if not torch.equal(output,actual):
                        raise ValueError('observer reconstruction differs from native layer output')
                    delta=actual.float()-baseline.float()
                    if not torch.isfinite(delta).all():raise ValueError('nonfinite native update')
                    row=self.rows[site]
                    row['squared_update_norm']+=float(delta.double().square().sum())
                    row['elements']+=delta.numel();row['calls']+=1
                    row['multi_token_calls' if args[0].shape[-2]>1 else 'single_token_calls']+=1
            self.handles.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]

    def report(self):
        if any(r['calls']<1 or r['elements']<1 for r in self.rows.values()):
            raise ValueError('incomplete projection-site coverage')
        return {name:dict(row) for name,row in self.rows.items()}


def trajectory_gate(reference,control,*,tolerance=.2):
    if not reference or len(reference)!=len(control) or tolerance!=.2:
        raise ValueError('paired trajectories and fixed 20 percent validation tolerance required')
    sites=set(reference[0]);reports={}
    if not sites or any(set(row)!=sites for row in reference+control):
        raise ValueError('trajectory site coverage differs')
    for site in sorted(sites):
        stats=[]
        for bank in (reference,control):
            values=[row[site] for row in bank]
            if any(not math.isfinite(v['squared_update_norm']) or v['squared_update_norm']<0
                   or type(v['elements']) is not int or v['elements']<1
                   or type(v['calls']) is not int or v['calls']<1 for v in values):
                raise ValueError('invalid trajectory energy or coverage')
            stats.append(dict(energy=sum(v['squared_update_norm'] for v in values),
                              elements=sum(v['elements'] for v in values),calls=sum(v['calls'] for v in values)))
        ref,ctrl=stats
        ratios={};passed=True
        for name,r,c in [('trajectory_l2',ref['energy'],ctrl['energy']),
                         ('pooled_rms',ref['energy']/ref['elements'],ctrl['energy']/ctrl['elements'])]:
            ratio=math.sqrt(c/r) if r else None
            good=(c==0) if r==0 else abs(ratio-1)<=tolerance
            ratios[name]=ratio;passed=passed and good
        reports[site]=dict(passed=passed,ratios=ratios,reference=ref,control=ctrl)
    return dict(passed=all(v['passed'] for v in reports.values()),sites=reports,
                tolerance=tolerance,paired_inputs=len(reference),
                measurement='per-site sum of native local-update energy over complete visited trajectories; also pooled per-element RMS',
                policy='all sites on both tasks and all four runs must pass; no fitted rescaling or successful-subset selection')
