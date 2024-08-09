#!/bin/bash
cd ../
DATASET_NAME="peermeta"

PLM_MODEL_PATH="/home/miao4/punim0521/ModularMetaReview/results/led_large_peermeta/checkpoints/checkpoint-1050"
LENGTH_INPUT=16384
LENGTH_TGT=512

python led_inference.py  \
                --dataset_name ${DATASET_NAME} \
                --save_path ../../results/led_large_${DATASET_NAME} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_beams 5 \
                --max_length_tgt ${LENGTH_TGT} \
                --max_length_source ${LENGTH_INPUT} \
                --num_test_data -1 \
                --no_repeat_ngram_size 3 \
                --length_penalty 0.8