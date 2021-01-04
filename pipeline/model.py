#!/usr/bin/env python3

import torch.nn as nn
from transformers import BertForSequenceClassification

class BERT(nn.Module):

    def __init__(self, model_name):
        super(BERT, self).__init__()

        self.model_name = model_name
        self.encoder = BertForSequenceClassification.from_pretrained(self.model_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea
