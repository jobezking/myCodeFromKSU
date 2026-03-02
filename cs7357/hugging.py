import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio

import wandb
from dotenv import load_dotenv

load_dotenv() # Loads the variables from .env
wandb_key  = os.getenv("WANDB_API_KEY")
token = os.getenv("HF_TOKEN")
wandb.login(key=wandb_key)
from huggingface_hub import login
login(token=HF_TOKEN)


from datasets import load_dataset

data_files = {
    'train': '/media/type3/data/train_dm.csv',
    'valid': '/media/type3/data/valid_dm.csv'
}

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t"
)

train_data = dataset['train']
valid_data = dataset['valid']

print(train_data)
print(valid_data)

repo_name = "wav2vec2-large-xls-r-300m-dm32"

input_col = 'path'
output_col = 'label'
audio_len = 32

label_list = train_data.unique(output_col)
label_list.sort()
num_classes = len(label_list)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_list}")