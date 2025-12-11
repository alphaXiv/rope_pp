#!/bin/bash
set -x

# Create necessary directories
mkdir -p checkpoints logs results wandb

export WANDB_RUN_GROUP="rope_pp-376m-single-gpu"

# Stage 1: Initial training for 100k steps
echo "=========================================="
echo "Starting Stage 1: Initial Training"
echo "=========================================="

python train_rope_pp_single_gpu.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr 'rope_pp-376m-4k-imag2-single-gpu' \
  > logs/rope_pp-376m-4k-imag2-single-gpu.log 2>&1

echo "Stage 1 complete! Check logs/rope_pp-376m-4k-imag2-single-gpu.log"

# Wait for stage 1 to complete
wait

# Stage 2: Learning rate decay for 10k steps from checkpoint 90000
echo "=========================================="
echo "Starting Stage 2: Learning Rate Decay"
echo "=========================================="

python train_rope_pp-decay_single_gpu.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr 'rope_pp-376m-4k-imag2-single-gpu' \
  --load_ckpt 90000 \
  --decay_step 10000 \
  > logs/rope_pp-376m-4k-imag2-single-gpu-ckpt90000-decay.log 2>&1

echo "Stage 2 complete! Check logs/rope_pp-376m-4k-imag2-single-gpu-ckpt90000-decay.log"

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final model saved at: checkpoints/rope_pp-376m-4k-imag2-single-gpu-ckpt90000-decay"
echo ""
echo "To monitor training progress, run:"
echo "  tail -f logs/rope_pp-376m-4k-imag2-single-gpu.log"
echo "  tail -f logs/rope_pp-376m-4k-imag2-single-gpu-ckpt90000-decay.log"
