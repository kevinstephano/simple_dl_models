import sys
import torch

from engines import runner

def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device) :
    return [[torch.randn(128, 32, 56, 56, dtype=dtype, device=device)] for _ in range(steps)]   

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.conv = torch.nn.Conv2d(32, 64, kernel_size=3, groups=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.act = torch.nn.ReLU()

    def forward(self, inputs) :
        out1 = self.conv(inputs)
        out2 = self.bn(out1)
        out3 = self.act(out2)
        return out3.sum()

if __name__ == "__main__" :
    runner.run(sys.argv, TestModule(), optim_func, input_func, None) 
