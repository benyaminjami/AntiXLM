# import dependencies
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import ankh
import re
import numpy as np
import pandas as pd
import copy

import transformers
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Model,
    AutoTokenizer,
)
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification

# from evaluate import load
# from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_n_mask_tokens(n):
    return [f"<extra_id_{i}>" for i in range(n)]


def append_n_mask_tokens(input_, n):
    return input_ + "".join(get_n_mask_tokens(n))


device = "cuda"
# model, tokenizer = ankh.load_base_model(generation=True)
model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
model.eval()
model.to(device=device)

test_seq = "QVQLVESGGGLVQPGGSL"
num_new_tokens = 5
masked_seq = append_n_mask_tokens(test_seq, n=num_new_tokens)
maximum_length = num_new_tokens * 2 + 1
num_beams = 5
temperature = 1.0
encoded = tokenizer.batch_encode_plus(
    [masked_seq, append_n_mask_tokens(test_seq, n=num_new_tokens + 2)],
    add_special_tokens=True,
    return_tensors="pt",
    padding=True,
)
input_ids = encoded["input_ids"].to(device)
input_ids

b = model(input_ids=input_ids, attention_mask=encoded['attention_mask'].to(device), labels=input_ids, return_dict=True)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    a = model(input_ids=input_ids, attention_mask=encoded['attention_mask'].to(device), labels=input_ids, return_dict=True)
a = m(
    input_ids=input_ids,
    attention_mask=encoded["attention_mask"].to(device),
    labels=input_ids,
    return_dict=True,
)
generation = model.generate(
    input_ids=input_ids,
    temperature=temperature,
    max_length=maximum_length + 10,
    return_dict_in_generate=True,
    return_dict=True,
    # decoder_input_ids=torch.tensor([[0,2,3], [0,2,3]]).to(device)
)

output_ids = generation[0].squeeze()
