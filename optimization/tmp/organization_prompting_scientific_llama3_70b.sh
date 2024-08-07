#!/bin/bash

DATASET_PATH="/home/miao4/punim0521/ModularMetaReview/annotations/scientific_reviews/annotation_data_small.json"
MODEL_NAME_OR_PATH="/data/projects/punim0521/tmp/llama3/Meta-Llama-3-70B-Instruct-four-nodes/"
MAX_LENGTH_GENERATION=1024

torchrun --nproc_per_node 4 organization_prompting_scientific_llama3_70b.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --max_length_model 8192 \
                --max_predict_length ${MAX_LENGTH_GENERATION} \
                --dataset_path ${DATASET_PATH} \
                --num_test_samples 5 \
                --top_p 0.95 \
                --output_file scientific_categorization_result_llama3_70b.json \