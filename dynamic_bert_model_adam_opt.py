import sys
import math
import random
import torch
from torch import nn

from execution import runner

from bert_model_adam_opt import BertForPreTraining
from bert_model_adam_opt import BertConfig
from bert_model_adam_opt import optim_func
from dynamic_bert_model import input_func

if __name__ == "__main__" :
    sys.argv.append('--grad_accum_steps=4')
    # We need a large number of warmup steps pre-compile all the variants
    # TODO: breakout the warmup steps and just iterate through the known sizes
    sys.argv.append('--warmup_steps=30')
    runner.run(sys.argv, 'Dynamic-BertModel-AdamOpt', BertForPreTraining(BertConfig()), optim_func, input_func, None) 
