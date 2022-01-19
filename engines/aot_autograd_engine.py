import torch

from functorch.compile import aot_module, partition_with_recompute_fwd_in_bwd, ts_compile, decomposition_table

def train_loop(args, model, optim_func, data_func, grad_func=None) :
    model.to(device=args.device)
    aot_model = aot_module(mod, ts_compile, partition_fn=partition_with_recompute_fwd_in_bwd, decompositions=decomposition_table)
   
    optimizer = optim_func(model.parameters())

    batches = data_func(args.steps, args.data_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.data_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    with torch.jit.fuser('fuser2') :
        for step,batch in enumerate(batches) :
            if step == args.warmup_steps :
                start_evt.record()

            loss = aot_model(*batch)
            if grads :
                loss.backward(grads)
            else :
                loss.backward()
 
            if step % args.grad_accum_steps == 0 :
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
    stop_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(stop_evt)
