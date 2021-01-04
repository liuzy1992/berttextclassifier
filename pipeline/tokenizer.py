#!/usr/bin/env python3

from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch
from . import device

def tokenizer(data_path, model_path, max_length, batch_size):
    tokenizer = BertTokenizer.from_pretrained(model_path)

    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True, fix_length=max_length, pad_token=pad_index, unk_token=unk_index)
    fields = [('label', label_field), ('title', text_field), ('content', text_field), ('titlecontent', text_field)]

    # TabularDataset
    train, valid, test = TabularDataset.splits(path=data_path, train='train.tsv', validation='valid.tsv',test='test.tsv', format='TSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.content), device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.content), device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)
    
    return train_iter, valid_iter, test_iter
