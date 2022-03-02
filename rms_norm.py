import sys
import torch

from engines import runner

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, dtype, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, dtype=dtype, device=device), requires_grad=True)
        self.variance_epsilon = 1e-6

    def forward(self, x : torch.Tensor):
        variance = (x * x).mean(-1, keepdim=True)
        x_hat = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_hat

def input_func(steps, dtype, device) :
    return [[torch.randn(24, 128, 1024, dtype=dtype, device=device)] for _ in range(steps)]   

def grad_func(steps, dtype, device) :
    return [torch.randn(24, 128, 1024, dtype=dtype, device=device) for _ in range(steps)]

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.norm = RMSNorm(1024, torch.float, 'cuda')

    def forward(self, inputs) :
        out1 = self.norm(inputs)
        return out1

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, TestModule(), optim_func, input_func, grad_func) 
