#!/bin/bash
cd ../
DATASET_NAME="peermeta"

MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-8B"
SAVE_NAME="llama31_8b"

torchrun --nnodes 1 --nproc_per_node=4 --master_port=9822 llama_finetune.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --dataset_path ../../datasets \
                --dataset_name ${DATASET_NAME} \
                --max_length_model 16384 \
                --max_predict_length 512 \
                --num_training_samples -1 \
                --keep_split -1 \
                --num_val_samples -1 \
                --num_test_samples -1 \
                --do_train True \
                --output_dir ../../results/${SAVE_NAME}_${DATASET_NAME} \
                --num_train_epochs 5 \
                --bf16 True \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps 1 \
                --gradient_checkpointing True \
                --evaluation_strategy "steps" \
                --eval_steps 500 \
                --save_strategy "steps" \
                --save_steps 500 \
                --save_total_limit 1 \
                --save_only_model True \
                --optim adafactor \
                --learning_rate 1e-6 \
                --warmup_ratio 0.2 \
                --label_smoothing_factor 0.1\
                --lr_scheduler_type "cosine" \
                --logging_steps 10 \
                --project_name llama31_8b_${DATASET_NAME} \
                --report_to "wandb" \
                --run_name ${SAVE_NAME}_${DATASET_NAME} \
                --overwrite_output_dir True \
                --fsdp "full_shard auto_wrap offload" \
                --fsdp_config "fsdp_config.json" \
                --ddp_timeout 14400
