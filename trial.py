import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline
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
from tqdm import tqdm
from datasets import Dataset, DatasetDict, ClassLabel, Value

sns.set(rc={'figure.figsize':(8, 6)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)

from huggingface_hub import notebook_login
notebook_login()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from trl import SFTTrainer
from trl import setup_chat_format

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import bitsandbytes as bnb
from datasets import DatasetDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict(test_data, model, tokenizer):
    y_pred = []
    pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 6, #4
                        temperature = 0.0,
                       )
    for i in tqdm(range(len(test_data))):
        prompt = test_data[i]["text"]
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
        print("Answer:", answer, "ANswer finished")
        if "2" in answer:
            y_pred.append(2)
        elif "0" in answer:
            y_pred.append(0)
        elif "1" in answer:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

def evaluate(y_true, y_pred):
    # convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=list(unique_labels))
    print('\nConfusion Matrix:')
    print(conf_matrix)

def generate_prompt(data_point):
    data_point['text'] = data_point['text'].lower()
    prompt = (f"""
        Analyze the sentiment of the movie review enclosed in square brackets, determine if it is positive(represented as 2), neutral(represented as 1), or negative (represented as 0), and return the answer as the corresponding sentiment label 0, 1, 2.
        [{data_point['text']}] = {data_point['sentiment']}""".strip())
    return {'text': prompt}

# prompt: "Analyze the sentiment of the movie review enclosed in square brackets, determine if it is positive, neutral, or negative, and return the answer as the corresponding sentiment label "positive" or "neutral" or "negative"."
def generate_test_prompt(data_point):
    data_point['text'] = data_point['text'].lower()
    prompt = (f"""
        Analyze the sentiment of the movie review enclosed in square brackets, determine if it is positive(represented as 2), neutral(represented as 1), or negative (represented as 0), and return the answer as the corresponding sentiment label 0, 1, 2.
        [{data_point['text']}]""".strip())
    return {'text': prompt}

# Data processing -> Prompt generation 
# load csv data
# data_path = "Sentiment_Dataset/only_data_cleaning/IMDB_Dataset_cleaned.csv"
data_path = "Sentiment_Dataset/only_data_cleaning/BTC_Dataset_cleaned.csv"
data = load_dataset('csv', data_files=data_path)
print(data)

data["train"] = data["train"].class_encode_column("sentiment")
print(data)
#Print mapping dictionary like for sentiment netural is 0, positive is 1 and negative is 2
# print("MAPPING: ",data["train"].features["sentiment"].str2int)
# Now you can split the dataset with stratification
train_val = data['train'].train_test_split(test_size=0.2, shuffle=True, seed=42, stratify_by_column='sentiment')
print(train_val["train"][0])
# print no of samples per class in train and test data
from collections import Counter
print("TRAIN DATA: ", Counter(train_val["train"]["sentiment"]))
print("TEST DATA: ", Counter(train_val["test"]["sentiment"]))

# Apply the map function to modify the 'text' column based on generate_prompt function
train_val['train'] = train_val['train'].map(generate_prompt, batched=False, remove_columns=['text', 'sentiment'])
train_val['test'] = train_val['test'].map(generate_test_prompt, batched=False, remove_columns=['text'])
train_data = train_val["train"]
test_data = train_val["test"] 
print(train_data)
print(test_data)

print("ONE SAMPLE: ", train_data[0], "TRAIN DATA SAMPLE FINISHED.")
print("ONE SAMPLE: ", test_data[0], "TEST DATA SAMPLE FINISHED.")


# Model Loading and Tokenization
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
MY_TOKEN = "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" 
CUTOFF_LEN = 2048

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = LlamaForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf", quantization_config=bnb_config, token=MY_TOKEN)
model.config.pretraining_tp = 1 #Need to verify
model.config.use_cache = False

tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", token=MY_TOKEN)
tokenizer.pad_token_id = (0) # unk. we want this to be different from the eos token
tokenizer.padding_side = "right"

model, tokenizer = setup_chat_format(model, tokenizer)

# Lora-Config and Model Preparation
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = "all-linear"

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_EPOCHS = 1
OUTPUT_DIR = "ckpts/sentiment_analysis/BTC_Dummy/"

training_arguments = transformers.TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_strategy="steps",
    logging_steps=10, #10
    learning_rate=LEARNING_RATE,
    fp16=True,
    max_steps=-1,
    warmup_ratio=0.03,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
)

# SFT TRAINER - INHERITS TRANSFORMERS.TRAINER but takes LORA CONFIG as well 
# Ref: "https://discuss.huggingface.co/t/when-to-use-sfttrainer/40998/4"
sft_trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=CUTOFF_LEN,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    }
)

# Training
sft_trainer.train()

# Save the model
sft_trainer.model.save_pretrained(OUTPUT_DIR)
# sft_trainer.model.push_to_hub("ML-Security/ALPACA-LORA-IMDB-SA", use_auth_token=True)

# Predictions
y_pred = predict(test_data, model, tokenizer)
y_true = test_data["sentiment"]
evaluate(y_true, y_pred)

# Save the evaluation
evaluation = pd.DataFrame({'text': test_data["text"], 
                           'y_true':y_true, 
                           'y_pred': y_pred},)
evaluation.to_csv(f"{OUTPUT_DIR}/evaluation.csv", index=False)