#!/bin/bash
cd ../
DATASET_NAME="peersum"

#PLM_MODEL_PATH="allenai/led-large-16384"
PLM_MODEL_PATH="/home/miao4/punim0521/NeuralAbstractiveSummarization/reproduced/led_summarization_huggingface/result_peersum_4096_512_paper_hyperparams/checkpoints/checkpoint-825"
LENGTH_INPUT=4096
LENGTH_TGT=512

python led_inference.py  \
                --dataset_name ${DATASET_NAME} \
                --save_path result_${DATASET_NAME}_${LENGTH_INPUT}_${LENGTH_TGT}_paper_hyperparams \
                --pretrained_model ${PLM_MODEL_PATH} \
                --num_beams 5 \
                --max_length_tgt ${LENGTH_TGT} \
                --max_length_source ${LENGTH_INPUT} \
                --num_test_data -1 \
                --no_repeat_ngram_size 3 \
                --length_penalty 0.8