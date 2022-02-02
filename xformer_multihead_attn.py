import sys
import math
import torch
from torch import nn
from torch.nn import functional as F

from engines import runner

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

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout_prob, num_attention_heads):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x : torch.Tensor, sequences : int):
        x = x.view(-1, sequences * self.num_attention_heads, self.attention_head_size).transpose(0,1)
        return x

    def transpose_key_for_scores(self, x : torch.Tensor, sequences : int):
        x = x.view(-1, sequences * self.num_attention_heads, self.attention_head_size).transpose(0,1).transpose(1,2)
        return x

    def forward(self, hidden_states, attention_mask):
        sequences = hidden_states.size(1)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, sequences)
        key_layer = self.transpose_key_for_scores(mixed_key_layer, sequences)
        value_layer = self.transpose_for_scores(mixed_value_layer, sequences)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer)
        attention_scores = attention_scores.unflatten(0, (sequences, self.num_attention_heads))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        attention_probs = attention_probs.flatten(0, 1)
        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.transpose(0,1).contiguous()
        context_layer = context_layer.view(-1, sequences, self.all_head_size)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TestModule(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(TestModule, self).__init__()
        self.self = BertSelfAttention(hidden_size, dropout_prob, num_attention_heads)
        self.output = BertSelfOutput(hidden_size, dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, TestModule(1024, 16, 0.1), optim_func, input_func, grad_func) 
