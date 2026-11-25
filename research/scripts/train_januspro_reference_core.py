#!/usr/bin/env python3
"""Isolated Janus-Pro entry point for the existing reference-core trainer."""
from scripts import train_cross_backbone_pilot as training
from skillbridge.januspro_runtime import load_training
from skillbridge.training_overlap import training_lock

def main():
    training.BACKBONES = ('januspro',)
    training.LOADERS['januspro'] = load_training
    args = training.parse_args()
    if (args.method, args.rank, args.timestep_buckets, args.image_size) != ('tied_core',64,1,384):
        raise ValueError('registered rank-64 native Janus-Pro recipe required')
    if not args.preserve_source_order or args.shuffle_source_labels:
        raise ValueError('ordered unmodified source records required')
    if tuple(args.target_suffixes) != ('q_proj','v_proj'):
        raise ValueError('reference q_proj/v_proj targets required')
    with training_lock(args.output):
        training.run_training(args)

if __name__=='__main__': main()
