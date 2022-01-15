import torch

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm

def train_loop(args, model, optim_func, data_func) :
    model.to(device=args.device)
   
    optimizer = optim_func(model.parameters())

    batches = data_func(args.steps, args.data_dtype, args.device)

    with torch.jit.fuser('fuser2') :
        for step,batch in enumerate(batches) :
            loss = model(batch)
            loss.backward()
 
            if step % args.grad_accum_steps == 0 :
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            ltm.mark_step()
