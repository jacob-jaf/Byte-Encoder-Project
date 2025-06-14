import torch.nn as nn
import torch
from math import sqrt

def GeLu():
    '''
    Implements gaussian error linear unit
    same as x * gaussian CDF(x)
    '''
    return x * 0.5 * (1 + torch.erf(x / sqrt(2)))

class nn_w_RMS(nn.Module):

    def __init__(self, hidden_layers)