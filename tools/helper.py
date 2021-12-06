import math
import torch
from torch import nn
from torch.autograd import grad as torch_grad
from random import random

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i