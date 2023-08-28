from gen_config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from math import sin, cos
from einops import rearrange, repeat



def init_weights(m):
    class_name=m.__class__.__name__

    if "Conv2d" in class_name or "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Linear(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, dataset): 
        super(Linear, self).__init__()
        if "ntu60" in dataset:
            label_num = 60
        elif "ntu120" in dataset:
            label_num = 120
        elif "pku" in dataset:
            label_num = 51
        else:
            raise ValueError
        self.classifier = nn.Linear(hidden_size, label_num)
        self.apply(init_weights)

    def forward(self, X):
        X = self.classifier(X)
        return X