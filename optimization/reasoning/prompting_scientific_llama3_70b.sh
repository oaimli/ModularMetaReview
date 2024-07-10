#!/bin/bash

DATASET_PATH="../selection/scientific_selection_result_llama3_70b.json"
MODEL_NAME_OR_PATH="/data/projects/punim0521/tmp/llama3/Meta-Llama-3-70B-Instruct-four-nodes/"
MAX_LENGTH_GENERATION=1024

torchrun --nproc_per_node 4 llama3_inference.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --max_length_model 8192 \
                --max_predict_length ${MAX_LENGTH_GENERATION} \
                --dataset_path ${DATASET_PATH} \
                --top_p 0.95 \
                --output_file scientific_reasoning_result_llama3_70b.json \