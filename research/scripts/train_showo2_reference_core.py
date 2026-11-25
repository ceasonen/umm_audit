#!/usr/bin/env python3
"""Isolated Show-o2 entry point for the existing reference-core trainer."""
from scripts import train_cross_backbone_pilot as training
from skillbridge.showo2_runtime import load_training
from skillbridge.training_overlap import training_lock

def main():
    training.BACKBONES = ('showo2',)
    training.LOADERS['showo2'] = load_training
    args = training.parse_args()
    if (args.method, args.rank, args.timestep_buckets, args.image_size) != ('tied_core',64,1,432):
        raise ValueError('registered rank-64 native Show-o2 recipe required')
    if not args.preserve_source_order or args.shuffle_source_labels:
        raise ValueError('ordered unmodified source records required')
    if tuple(args.target_suffixes) != ('q_proj','v_proj'):
        raise ValueError('reference q_proj/v_proj targets required')
    with training_lock(args.output):
        training.run_training(args)

if __name__=='__main__': main()
