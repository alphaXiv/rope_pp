#!/bin/bash

set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-376m"

wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp.py --config_abbr '376m' --imag --imag_mode 'imag2' \
--save_abbr 'rope_pp-376m-4k-imag2' > logs/rope_pp-376m-4k-imag2.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-decay.py --config_abbr '376m' --imag --imag_mode 'imag2' \
--save_abbr 'rope_pp-376m-4k-imag2' --load_ckpt 90000 --decay_step 10000 > logs/rope_pp-376m-4k-imag2-ckpt90000-decay.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx.py --config_abbr '376m' --imag --imag_mode 'imag2' \
--save_abbr 'rope_pp-376m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-376m-4k-imag2-ckpt90000-decay-lctx.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx-linear.py --config_abbr '376m' --factor 8  --imag --imag_mode 'imag2' \
--save_abbr 'rope_pp-376m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-376m-4k-imag2-ckpt90000-decay-lctx-pi8.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx-linear.py --config_abbr '376m' --yarn --factor 32 --imag --imag_mode 'imag2' \
--save_abbr 'rope_pp-376m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-376m-4k-imag2-ckpt90000-decay-lctx-yarn32.log 2>&1
