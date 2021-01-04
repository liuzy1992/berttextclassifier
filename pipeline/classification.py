#!/usr/bin/env python3

import torch
import pandas as pd
from transformers import BertTokenizer
from torchtext.data import Field, Iterator
from . import device
from .preprocessing import trim_string
from .savingandloading import load_checkpoint
from .model import BERT


def preprocessing_newdata(infile):
    df_data = pd.read_csv(infile, sep='\t', header=0)
    df_data['titlecontent'] = df_data['title'] + ". " + df_data['content']
    
    df_data['content'] = df_data['content'].apply(trim_string)
    df_data['titlecontent'] = df_data['titlecontent'].apply(trim_string)

    return df_data

def tokenizing_newdata(df, model_path, max_length, batch_size):
    tokenizer = BertTokenizer.from_pretrained(model_path)

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True)
    fields = [('id', None), ('title', text_field), ('content', text_field), ('titlecontent', text_field)]

    data_ite
