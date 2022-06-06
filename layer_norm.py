import sys
import torch

from execution import runner

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        u = x.mean(2, keepdim=True) # Specifying -1 for reduction dimension causes an error in TorchScript
        s = (x - u)
        s = s * s
        s = s.mean(2, keepdim=True) # Specifying -1 for reduction dimension causes an error in TorchScript
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

def input_func(steps, dtype, device) :
    return [[torch.randn(128, 128, 1024, dtype=dtype, device=device)] for _ in range(steps)]   

def grad_func(steps, dtype, device) :
    return [torch.randn(128, 128, 1024, dtype=dtype, device=device) for _ in range(steps)]

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.ln = BertLayerNorm(1024)

    def forward(self, inputs) :
        out1 = self.ln(inputs)
        return (out1,)

from components.dummy_optimizer import optim_func

if __name__ == "__main__" :
    runner.run(sys.argv, 'LayerNorm', TestModule(), optim_func, input_func, grad_func) 
