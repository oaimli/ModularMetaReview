#!/bin/bash
cd ../llama_topical_chat
DATASET_NAME="topical_chat"

MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
SAVE_NAME="llama_7b"

torchrun --nnodes 1 --nproc_per_node=4 --master_port=9822 llama_finetune.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --dataset_path ../../dataset \
                --dataset_name ${DATASET_NAME} \
                --max_length_model 1280 \
                --max_predict_length 128 \
                --num_training_samples -1 \
                --keep_split 2 \
                --num_val_samples 512 \
                --num_test_samples 512 \
                --do_train True \
                --output_dir ../../result/${SAVE_NAME}_${DATASET_NAME}_02 \
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
                --optim adafactor \
                --learning_rate 1e-6 \
                --warmup_ratio 0.2 \
                --label_smoothing_factor 0.1\
                --lr_scheduler_type "cosine" \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --metric_for_best_model eval_loss \
                --greater_is_better False \
                --project_name "FGFT" \
                --report_to "wandb" \
                --run_name ${SAVE_NAME}_${DATASET_NAME}_02 \
                --overwrite_output_dir True \
                --tf32 True \
                --fsdp "full_shard auto_wrap offload" \
                --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
                --ddp_timeout 14400
