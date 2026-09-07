#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export DIFFSYNTH_MODEL_BASE_PATH=/root/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_9B="black-forest-labs/FLUX.2-klein-9B"
CACHE="/root/train_9b_v1/cache"
OUT="/root/train_9b_v1"

mkdir -p "$OUT"

MODULES="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj"
for i in $(seq 0 23); do
    MODULES="$MODULES,single_transformer_blocks.${i}.attn.to_out"
done

echo "============================================"
echo "9B Phase 2: LoRA Training (V1 config, rank=64)"
echo "============================================"

cd /root/DiffSynth-Studio
accelerate launch examples/flux2/model_training/train.py \
    --dataset_base_path "$CACHE" \
    --max_pixels 1048576 \
    --dataset_repeat 2 \
    --model_id_with_origin_paths "${MODEL_9B}:transformer/*.safetensors" \
    --tokenizer_path "${MODEL_9B}:tokenizer/" \
    --learning_rate 1e-4 \
    --num_epochs 35 \
    --save_steps 5200 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$OUT" \
    --lora_base_model dit \
    --lora_rank 64 \
    --lora_target_modules "$MODULES" \
    --use_gradient_checkpointing \
    --task sft:train

echo ""
echo "Phase 2 COMPLETE"
echo "Output: $OUT"
