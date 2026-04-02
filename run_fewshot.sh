#!/bin/bash

CKPT="./experiments/pretrain_modelnet_v3/ckpt-last.pth"
CONFIG="./cfgs/fewshot.yaml"

for way in 5 10; do
    for shot in 10 20; do
        for fold in $(seq 0 9); do
            python3 fewshot.py \
                --config $CONFIG \
                --ckpts  $CKPT \
                --exp_name fewshot_${way}way${shot}shot_v4 \
                --way  $way \
                --shot $shot \
                --fold $fold
        done
    done
done