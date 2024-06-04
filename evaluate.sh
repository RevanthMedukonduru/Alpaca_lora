#!/bin/bash

CUDA_VIBILE_DEVICES=1 python evaluate.py \
    --load_8bit \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --lora_weights 'ML-Security/Alpaca-Lora-Twitter-Sentiment_analysis' \
    --prompt_template 'sentiment_alpaca' \
    --file_path 'Sentiment_Dataset/processed_data/cleaned-Reddit-Data.json'