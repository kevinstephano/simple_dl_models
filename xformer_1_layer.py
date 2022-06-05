import sys
import math
import torch
from torch import nn

from execution import runner

import xformer_multihead_attn
import xformer_feed_fwd

def input_func(steps, dtype, device) :
    results = []
    for _ in range(steps) :
        data = torch.randn(128, 64, 1024, dtype=dtype, device=device)
        mask = torch.randn(64, 1, 1, 128, dtype=dtype, device=device)
        bool_mask = mask < 0.
        results.append([data, bool_mask])
    return results

def grad_func(steps, dtype, device) :
    return [torch.randn(128, 64, 1024, dtype=dtype, device=device) for _ in range(steps)]

class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout_prob):
        super(BertLayer, self).__init__()
        self.self = xformer_multihead_attn.BertSelfAttention(hidden_size, dropout_prob, num_attention_heads)
        self.self_output = xformer_multihead_attn.BertSelfOutput(hidden_size, dropout_prob)
        self.intermediate = xformer_feed_fwd.BertIntermediate(hidden_size, intermediate_size)
        self.output = xformer_feed_fwd.BertOutput(hidden_size, intermediate_size, dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.self_output(self_output, input_tensor)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,)

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, "Transformer-1-Layer", BertLayer(1024, 4096, 16, 0.1), optim_func, input_func, grad_func) 
