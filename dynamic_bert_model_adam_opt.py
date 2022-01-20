import sys
import math
import random
import torch
from torch import nn

from engines import runner

from bert_model_adam_opt import BertForPreTraining
from bert_model_adam_opt import BertConfig
from bert_model_adam_opt import optim_func
from dynamic_bert_model import input_func

if __name__ == "__main__" :
    runner.run(sys.argv, BertForPreTraining(BertConfig()), optim_func, input_func, None) 
