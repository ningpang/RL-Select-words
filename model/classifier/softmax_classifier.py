import torch
import torch.nn as nn
from ..base_model import base_model
class Softmax_Layer(base_model):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=True)


    def forward(self, input):
        logits = self.fc(input)
        return logits