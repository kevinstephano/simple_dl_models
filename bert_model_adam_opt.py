import sys
import math
import torch
from torch import nn

from engines import runner

def optim_func(params) :
    return torch.optim.AdamW(params)

from bert_model import BertForPreTraining
from bert_model import BertConfig
from bert_model import data_func

if __name__ == "__main__" :
    runner.run(sys.argv, BertForPreTraining(BertConfig()), optim_func, data_func, None) 
