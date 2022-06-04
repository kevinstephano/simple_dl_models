import sys
import math
import torch
from torch import nn

from execution import runner

from apex.optimizers.fused_adam import FusedAdam

def input_func(steps, dtype, device) :
    vocab_size = 30528
    sequences = 64
    sequence_length = 128
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        segment_ids = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        input_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        masked_lm_labels = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        next_sentence_labels = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels])
    return results

def optim_func(params) :
    return FusedAdam(params)

if __name__ == "__main__" :
    config = BertConfig()
    config.num_hidden_layers = 1
    runner.run(sys.argv, 'HF-BertForPreTraining', BertForPreTraining(config), optim_func, input_func, None) 
