import sys
import torch
import random

from execution import runner

def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device) :
    max_seq_length = 128
    min_seq_length = 2
    seq_lengths = [random.randint(min_seq_length, max_seq_length) for _ in range(steps)]
    return [[torch.randn(128, seql, 1024, dtype=dtype, device=device)] for seql in seq_lengths]   

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(1024, 1024)
        self.act = torch.nn.ReLU()

    def forward(self, inputs) :
        out1 = self.linear(inputs)
        out2 = self.act(out1)
        out3 = out2 + inputs
        return out3.sum()

if __name__ == "__main__" :
    runner.run(sys.argv, 'Dynamic-Simple-Model', TestModule(), optim_func, input_func, None) 
