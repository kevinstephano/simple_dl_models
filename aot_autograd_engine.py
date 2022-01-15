import torch

from functorch.compile import aot_module, partition_with_recompute_fwd_in_bwd, ts_compile, decomposition_table

def train_loop(args, model, optim_func, data_func) :
    model.to(device=args.device)
    aot_model = aot_module(mod, ts_compile, partition_fn=partition_with_recompute_fwd_in_bwd, decompositions=decomposition_table)
   
    optimizer = optim_func(model.parameters())

    batches = data_func(args.steps, args.data_dtype, args.device)

    with torch.jit.fuser('fuser2') :
        for step,batch in enumerate(batches) :
            loss = aot_model(batch)
            loss.backward()
 
            if step % args.grad_accum_steps == 0 :
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
