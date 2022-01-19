import sys
import torch

from engines import runner

def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def data_func(steps, dtype, device) :
    return [[torch.randn(128, 1024, dtype=dtype, device=device, requires_grad=True)] for _ in range(steps)]   

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
    runner.run(sys.argv, TestModule(), optim_func, data_func, None) 
