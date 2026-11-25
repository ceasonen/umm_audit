#!/usr/bin/env python3
"""Train the new DiMOO cohort with fixed native preprocessing and saved batches."""
import json
from pathlib import Path

from skillbridge.dimoo_replication import audit_native_batch, preprocessing_rng


def main():
    import torch
    from transformers import AutoTokenizer
    from scripts import train_cross_backbone_pilot as training
    from skillbridge.training_overlap import training_lock
    from skillbridge.mechanism_sequence import lock_json
    from scripts.run_dimoo_protocol_audit import write
    args=training.parse_args()
    if (args.backbone,args.method,args.rank,args.timestep_buckets,args.image_size)!=(
            'dimoo','tied_core',64,1,256) or not args.preserve_source_order or args.shuffle_source_labels:
        raise ValueError('registered DiMOO reference parameterization required')
    out=args.output.resolve(); loader=training.LOADERS['dimoo']
    def audited_loader(**kwargs):
        with preprocessing_rng(kwargs['seed']):
            model,route,batches,objective=loader(**kwargs)
        tokenizer=AutoTokenizer.from_pretrained(str(kwargs['root']/'models/dimoo/Lumina-DiMOO'),trust_remote_code=True)
        records=kwargs['records']; audit=[]; stored=[]
        if len(records)!=len(batches):raise ValueError('one native batch per source record required')
        for record,batch in zip(records,batches):
            x=batch.model_inputs['input_ids']; y=batch.model_inputs['labels']
            if x.ndim!=2 or x.shape[0]!=1:raise ValueError('native batch size differs')
            prompt=f"<system>{record['system_prompt']}</system><user>{record['user_prompt']}</user>"
            encode=lambda text: tokenizer(text,truncation=True,max_length=1024,padding=False)['input_ids']
            result=audit_native_batch(x[0].tolist(),y[0].tolist(),prompt_tokens=encode(prompt),
                                     answer_tokens=encode(record['answer_text']+'</answer>'))
            audit.append(dict(split_key=record['split_key'],**result))
            stored.append(dict(split_key=record['split_key'],input_ids=x.cpu(),labels=y.cpu()))
        torch.save(stored,out/'native-training-batches.pt')
        write(out/'native-supervision-audit.json',dict(records=len(records),batches=audit,
            answer_tokens=sum(r['answer_tokens'] for r in audit),
            padding_tokens=sum(r['padding_tokens'] for r in audit),native_objective_unchanged=True))
        return model,route,batches,objective
    training.LOADERS['dimoo']=audited_loader
    with training_lock(out):
        lock_json(out/'native-protocol.json',dict(crop_and_mask_seed=args.seed,image_size=256,
            native_answer_span=1024,native_padding_supervision=True,training_precision='bfloat16',
            timestep_buckets=1,mask_schedule='official processor, one fixed batch per ordered example',
            input_and_label_tensors='native-training-batches.pt',all_outcomes_retained=True))
        training.run_training(args)


if __name__=='__main__':main()
