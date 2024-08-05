#!/bin/bash
cd ../llama_topical_chat
DATASET_NAME="topical_chat"

MODEL_NAME_OR_PATH=""
SAVE_NAME="llama_7b"


MAX_LENGTH_GENERATION=128
MIN_LENGTH_GENERATION=1


python llama_inference.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --max_length_model 1280 \
                --max_predict_length ${MAX_LENGTH_GENERATION} \
                --min_predict_length ${MIN_LENGTH_GENERATION} \
                --dataset_path ../../dataset/ \
                --dataset_name ${DATASET_NAME} \
                --num_test_samples -1 \
                --bf16 True \
                --do_sample True \
                --num_beams 5 \
                --top_p 0.95 \
                --do_predict True \
                --output_dir ../../result/${SAVE_NAME}_${DATASET_NAME} \
                --output_file generated_responses \
                --tf32 True