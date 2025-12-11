#!/bin/bash
set -x

# Create necessary directories
mkdir -p checkpoints logs results wandb

# Generate random port for deepspeed
port=$(shuf -i25000-30000 -n1)

# Experiment name used for checkpoints and logs
EXPERIMENT_NAME="rope_pp-376m-4k-imag2"

export WANDB_RUN_GROUP="rope_pp-376m"

# Stage 1: Initial training for 100k steps
echo "=========================================="
echo "Starting Stage 1: Initial Training"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  train_rope_pp.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "$EXPERIMENT_NAME" \
  > logs/${EXPERIMENT_NAME}.log 2>&1

echo "Stage 1 complete! Check logs/${EXPERIMENT_NAME}.log"

# Wait for stage 1 to complete
wait

# Stage 2: Learning rate decay for 10k steps from checkpoint 90000
echo "=========================================="
echo "Starting Stage 2: Learning Rate Decay"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  train_rope_pp-decay.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "$EXPERIMENT_NAME" \
  --load_ckpt 90000 \
  --decay_step 10000 \
  > logs/${EXPERIMENT_NAME}-ckpt90000-decay.log 2>&1

echo "Stage 2 complete! Check logs/${EXPERIMENT_NAME}-ckpt90000-decay.log"

# Wait for stage 2 to complete
wait

# Stage 3: Long context training from checkpoint 10000
echo "=========================================="
echo "Starting Stage 3: Long Context Training"
echo "=========================================="

deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
  train_rope_pp-lctx.py \
  --config_abbr '376m' \
  --imag \
  --imag_mode 'imag2' \
  --save_abbr "${EXPERIMENT_NAME}-ckpt90000-decay" \
  --load_ckpt 10000 \
  > logs/${EXPERIMENT_NAME}-ckpt90000-decay-lctx.log 2>&1

echo "Stage 3 complete! Check logs/${EXPERIMENT_NAME}-ckpt90000-decay-lctx.log"

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final model saved at: checkpoints/${EXPERIMENT_NAME}-ckpt90000-decay-lctx"
echo ""
echo "To monitor training progress, run:"
echo "  tail -f logs/${EXPERIMENT_NAME}.log"
echo "  tail -f logs/${EXPERIMENT_NAME}-ckpt90000-decay.log"
echo "  tail -f logs/${EXPERIMENT_NAME}-ckpt90000-decay-lctx.log"
