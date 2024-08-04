#!/bin/bash
cd ../
DATASET_NAME="peersum"

PLM_MODEL_PATH="allenai/led-large-16384"
LENGTH_INPUT=16384
LENGTH_TGT=512

python led_finetune.py  \
                --batch_size 1 \
                --dataset_name ${DATASET_NAME} \
                --save_path result_${DATASET_NAME}_${LENGTH_INPUT}_${LENGTH_TGT} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --gradient_checkpointing \
                --beam_size 5 \
                --total_steps 50000 \
                --accum_data_per_step 2 \
                --val_check_interval 50 \
                --max_length_tgt ${LENGTH_TGT} \
                --max_length_input ${LENGTH_INPUT} \
                --num_train_data 256 \
                --num_val_data 256 \
                --num_test_data 512 \
                --label_smoothing_factor 0.1 \
                --no_repeat_ngram_size 3 \
                --length_penalty 0.6 \
                --ealy_stopping_patience 3