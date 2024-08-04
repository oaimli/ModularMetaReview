#!/bin/bash
cd ../
DATASET_NAME="peersum"
#DATASET_NAME="peersum_with_disagreements"

PLM_MODEL_PATH="allenai/led-large-16384"
#LENGTH_INPUT=16384
LENGTH_INPUT=1024
LENGTH_TGT=512

python led_inference.py  \
                --save_path result_${DATASET_NAME}_${LENGTH_INPUT}_${LENGTH_TGT}_zeroshot_paper_hyperparams \
                --dataset_name ${DATASET_NAME} \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_beams 5 \
                --max_length_tgt ${LENGTH_TGT} \
                --max_length_source ${LENGTH_INPUT} \
                --num_test_data -1 \
                --no_repeat_ngram_size 3 \
                --length_penalty 0.8