#!/bin/bash

input_npy="/data/workspaces/jp/LLMs/normalised_chatml_rwkvready/summarisation.txt.20B.npy"
model_path="/data/workspaces/jp/LLMs/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth"
outdir=out_v01

     # --precision bf16 \
python train.py \
     --load_model "$model_path" \
     --proj_dir "out" \
     --data_file "$input_npy" \
     --data_type "numpy"  \
     --vocab_size 50277 \
     --ctx_len 8192 \
     --epoch_steps 10 \
     --epoch_count 1000 \
     --epoch_begin 0 \
     --epoch_save 5 \
     --micro_bsz 3 \
     --n_layer 40 \
     --n_embd 5120 \
     --pre_ffn 0 \
     --head_qk 0 \
     --lr_init 1e-4 \
     --lr_final 1e-4 \
     --warmup_steps 0 \
     --beta1 0.9 \
     --beta2 0.999 \
     --adam_eps 1e-8 \
     --accelerator gpu \
     --devices 8 \
     --precision bf16 \
     --strategy deepspeed_stage_2_offload \
     --grad_cp 1 \
     --lora \
     --lora_r 8 \
     --lora_alpha 16 \
     --lora_dropout 0.01 \
     --lora_parts att,ffn,time,ln
