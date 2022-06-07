import torch
from apex.optimizers.fused_adam import FusedAdam

# This is just going to set the gradients to None
# such that gradient accumulation is not triggered.
def optim_func(params) :
    class ApexOptimizer(FusedAdam) :
        def __init__(self, params) :
            self.params = params
            super(ApexOptimizer, self).__init__(params)

        def zero_grad(self, set_to_none=False) :
            self.set_grad_none=True
            super(ApexOptimizer, self).zero_grad()

    return ApexOptimizer(params)
