#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export DIFFSYNTH_MODEL_BASE_PATH=/root/models
export DIFFSYNTH_SKIP_DOWNLOAD=true

MODEL_9B="black-forest-labs/FLUX.2-klein-9B"
DATA_DIR="/root/datasets/dataA_1/dataA_1"
META="/root/train_9b_v1/meta.json"
CACHE="/root/train_9b_v1/cache"

echo "[1/3] Metadata..."
cd /root
python3 /root/prep_dataset.py --dataset "$DATA_DIR" --out "$META"

MODULES="to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj"
for i in $(seq 0 23); do
    MODULES="$MODULES,single_transformer_blocks.${i}.attn.to_out"
done

echo "[2/3] Phase 1: cache..."
cd /root/DiffSynth-Studio
accelerate launch examples/flux2/model_training/train.py \
    --dataset_base_path "$DATA_DIR" \
    --dataset_metadata_path "$META" \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --max_pixels 1048576 \
    --dataset_repeat 2 \
    --model_id_with_origin_paths "${MODEL_9B}:text_encoder/*.safetensors,${MODEL_9B}:vae/diffusion_pytorch_model.safetensors" \
    --fp8_models "${MODEL_9B}:text_encoder/*.safetensors" \
    --tokenizer_path "${MODEL_9B}:tokenizer/" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$CACHE" \
    --lora_base_model dit \
    --lora_rank 64 \
    --lora_target_modules "$MODULES" \
    --use_gradient_checkpointing \
    --task sft:data_process

echo "[3/3] Phase 1 COMPLETE"
echo "Cache: $CACHE"
