import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import base_model
from transformers import BertModel, BertConfig

class Policy_Net(base_model):
    def __init__(self, config):
        super(Policy_Net, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension of output
        self.output_size = config.encoder_output_size

        self.fc = nn.Linear(self.output_size, 2)

        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding method!')

        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size+config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([self.output_size])


    def forward(self, inputs):
        output = self.encoder(inputs)[0]
        logits = self.fc(output)

        return F.softmax(logits, dim=-1)




