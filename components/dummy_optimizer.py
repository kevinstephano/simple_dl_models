import torch

# This is just going to set the gradients to None
# such that gradient accumulation is not triggered.
def optim_func(params) :
    class DummyOptimizer :
        def __init__(self, params) :
            self.params = params
            # Tell the GradScaler to skip attempting to unscale
            self._step_supports_amp_scaling = True
            self._dummy = True

        @torch.no_grad()
        def step(self, closure=None, grad_scaler=None) :
            pass

        def zero_grad(self, set_to_none=False) :
            for p in self.params :
                if set_to_none :
                    p.grad = None
                else :
                    if p.grad != None :
                        p.grad.zero_()

    return DummyOptimizer(params)
