import sys
import math
import torch
from torch import nn

from engines import runner

from bert_model import BertForPreTraining
from bert_model import BertConfig
from bert_model import input_func

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    config = BertConfig()
    config.num_hidden_layers = 1
    runner.run(sys.argv, BertForPreTraining(config), optim_func, input_func, None) 
