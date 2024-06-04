import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
from utils.prompter import Prompter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import fire
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
import json

import numpy as np
from bigmodelvis import Visualization

sns.set(rc={'figure.figsize':(8, 6)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)

from huggingface_hub import notebook_login
notebook_login()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data = load_dataset("json", data_files="Sentiment_Dataset/processed_data/cleaned-IMDB-Data.json")
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
MY_TOKEN = "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" 
CUTOFF_LEN = 2048

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.pretraining_tp = 1

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, token=MY_TOKEN)
tokenizer.pad_token_id = (0) # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"

def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = """.strip()

def generate_and_tokenize_prompt(data_point, CUTOFF_LEN=2048, add_eos_token=True):
    print("datapoint:", data_point)

    prompt_template = "sentiment_alpaca"
    prompter = Prompter(prompt_template)
    
    # Prompt generation
    prompt = prompter.generate_prompt(data_point["input"], str(data_point["output"]))
    # print("PROMPT:", prompt)
    
    # Tokenization
    tokenized_full_prompt = tokenizer(prompt, return_tensors=None, padding=False, max_length=CUTOFF_LEN, truncation=True)
    
    if (
        tokenized_full_prompt["input_ids"][-1] != tokenizer.eos_token_id
        and len(tokenized_full_prompt["input_ids"]) < CUTOFF_LEN and add_eos_token
    ):
        tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
        tokenized_full_prompt["attention_mask"].append(1)
    
    tokenized_full_prompt["labels"] = tokenizer.encode(str(data_point["output"]), return_tensors=None, padding=False, max_length=CUTOFF_LEN, truncation=True)
    print(tokenized_full_prompt["labels"])
    
    if (
        tokenized_full_prompt["labels"][-1] != tokenizer.eos_token_id
        and len(tokenized_full_prompt["labels"]) < CUTOFF_LEN and add_eos_token
    ):
        tokenized_full_prompt["labels"].append(tokenizer.eos_token_id)
    
    return tokenized_full_prompt
    

train_val = data["train"].train_test_split(
    test_size=0.2, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)
print(f"Train size: {len(train_data)} | Test size: {len(val_data)}")

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "linear"
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 600 #300
OUTPUT_DIR = "experiments/sentiment_analysis/IMDb/"

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
Visualization(model).structure_graph()
model.print_trainable_parameters()


# Training, labels are already tokenized in input prompt in tokenized_full_prompt["labels"]
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10, #10
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50, #50
    save_steps=50, #50
    output_dir=OUTPUT_DIR,
    save_total_limit=6,
    load_best_model_at_end=True,
    report_to="tensorboard",
    label_smoothing_factor=0.1,
)

data_collator = transformers.DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict

trainer.train()
model.save_pretrained(OUTPUT_DIR)
model.push_to_hub("ML-Security/ALPACA-LORA-IMDB-SA", use_auth_token=True)