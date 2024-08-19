#!/bin/bash
cd ../
DATASET_NAME="peermeta"

#MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-8B"
MODEL_NAME_OR_PATH="/home/miao4/punim0521/ModularMetaReview/results/llama31_8b_peermeta/checkpoint-15500"
SAVE_NAME="llama31_8b"


MAX_LENGTH_GENERATION=512
MIN_LENGTH_GENERATION=1


python llama_inference.py  \
                --model_name_or_path ${MODEL_NAME_OR_PATH} \
                --max_length_model 16384 \
                --max_predict_length ${MAX_LENGTH_GENERATION} \
                --min_predict_length ${MIN_LENGTH_GENERATION} \
                --dataset_path ../../datasets/ \
                --dataset_name ${DATASET_NAME} \
                --num_test_samples -1 \
                --bf16 True \
                --do_sample True \
                --num_beams 5 \
                --top_p 0.95 \
                --do_predict True \
                --output_dir ../../results/${SAVE_NAME}_${DATASET_NAME} \
                --output_file generated_summaries \
                --tf32 True