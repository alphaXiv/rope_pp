#!/bin/bash

set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-1_5b"

wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp.py --config_abbr '1_5b' --imag --imag_mode 'imag1' \
--save_abbr 'rope_pp-1_5b-4k-imag1' > logs/rope_pp-1_5b-4k-imag1.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx.py --config_abbr '1_5b' --imag --imag_mode 'imag1' \
--save_abbr 'rope_pp-1_5b-4k-imag1' --load_ckpt 100000 > logs/rope_pp-1_5b-4k-imag1-ckpt100000-lctx.log 2>&1
