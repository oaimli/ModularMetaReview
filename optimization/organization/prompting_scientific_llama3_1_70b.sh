#!/bin/bash

DATASET_PATH="/home/miao4/punim0521/ModularMetaReview/annotations/scientific_reviews/annotation_data_small.json"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"
MAX_LENGTH_GENERATION=1024

torchrun --nproc_per_node 4 llama3_1_inference.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --max_length_model 8192 \
                --max_predict_length ${MAX_LENGTH_GENERATION} \
                --dataset_path ${DATASET_PATH} \
                --num_test_samples 5 \
                --output_file scientific_categorization_result_llama3_1_70b.json \