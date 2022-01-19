import torch

# This is just going to set the gradients to None
# such that gradient accumulation is not triggered.
def optim_func(params) :
    class DummyOptimizer :
        def __init__(self, params) :
            self.params = params

        def step(self) :
            pass

        def zero_grad(self, set_to_none=False) :
            for p in self.params :
                if set_to_none :
                    p.grad = None
                else :
                    if p.grad != None :
                        p.grad.zero_()

    return DummyOptimizer(params)
