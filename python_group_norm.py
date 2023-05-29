import sys
import torch

from execution import runner

class PythonGroupNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(PythonGroupNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = torch.nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

def input_func(steps, dtype, device) :
    return [[torch.randn(256, 128, 28, 28, dtype=dtype, device=device)] for _ in range(steps)]   

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.gn = PythonGroupNorm(256, 32)

    def forward(self, inputs) :
        out1 = self.gn(inputs)
        return (out1,)

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, 'PythonGroupNorm', TestModule(), optim_func, input_func, None) 
