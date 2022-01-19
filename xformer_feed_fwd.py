import sys
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from engines import runner

def data_func(steps, dtype, device) :
    results = []
    for _ in range(steps) :
        data = torch.randn(128, 64, 1024, dtype=dtype, device=device)
        results.append([data])
    return results

def grad_func(steps, dtype, device) :
    return [torch.randn(128, 64, 1024, dtype=dtype, device=device) for _ in range(steps)]

class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module.
    """
    def __init__(self, in_features, out_features, act=torch.nn.functional.gelu, bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None
        self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.act_fn(F.linear(input, self.weight, self.bias))

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TestModule(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super(TestModule, self).__init__()
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout_prob)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, TestModule(1024, 4096, 0.1), optim_func, data_func, grad_func) 
