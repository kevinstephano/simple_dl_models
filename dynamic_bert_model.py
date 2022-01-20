import sys
import math
import random
import torch
from torch import nn

from engines import runner

def input_func(steps, dtype, device) :
    vocab_size = 30528
    sequences = 64
    max_sequence_length = 128
    results = []
    for _ in range(steps) :
        sequence_length = random.randint(1, max_sequence_length // 8) * 8
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        segment_ids = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        input_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        masked_lm_labels = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        next_sentence_labels = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels])
    return results

from bert_model import BertForPreTraining
from bert_model import BertConfig
from bert_model import optim_func

if __name__ == "__main__" :
    sys.argv.append('--grad_accum_steps=4')
    runner.run(sys.argv, BertForPreTraining(BertConfig()), optim_func, input_func, None) 
