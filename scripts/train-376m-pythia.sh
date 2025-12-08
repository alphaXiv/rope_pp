#!/bin/bash

set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-376m"

wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_pythia.py  --config_abbr '376m' \
--save_abbr 'rope_pp-376m-4k-pythia' > logs/rope_pp-376m-4k-pythia.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_pythia-decay.py --config_abbr '376m' \
--save_abbr 'rope_pp-376m-4k-pythia' --load_ckpt 90000 --decay_step 10000 > logs/rope_pp-376m-4k-pythia-ckpt90000-decay.log 2>&1
