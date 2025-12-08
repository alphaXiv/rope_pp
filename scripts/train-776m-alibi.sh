#!/bin/bash

set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-776m"

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_alibi.py  --config_abbr '776m' \
--save_abbr 'rope_pp-776m-4k-alibi' > logs/rope_pp-776m-4k-alibi.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_alibi-decay.py --config_abbr '776m' \
--save_abbr 'rope_pp-776m-4k-alibi' --load_ckpt 90000 --decay_step 10000 > logs/rope_pp-776m-4k-alibi-ckpt90000-decay.log 2>&1

