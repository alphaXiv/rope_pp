#!/bin/bash

set -x
port=$(shuf -i25000-30000 -n1)
wait 
export WANDB_RUN_GROUP="rope_pp-776m"

wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_fope.py --config_abbr '776m' \
--save_abbr 'rope_pp-776m-4k-fope' > logs/rope_pp-776m-4k-fope.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_fope-decay.py --config_abbr '776m' \
--save_abbr 'rope_pp-776m-4k-fope' --load_ckpt 90000 --decay_step 10000 > logs/rope_pp-776m-4k-fope-ckpt90000-decay.log 2>&1
