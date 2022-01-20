import torch

from functorch.compile import aot_module, partition_with_recompute_fwd_in_bwd, ts_compile, decomposition_table

def train_loop(args, model, optim_func, input_func, grad_func=None) :
    model.to(device=args.device)
    model.to(dtype=args.model_dtype)
   
    optimizer = optim_func(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=(args.grad_scaler and not hasattr(optimizer, '_dummy')))
    
    aot_model = aot_module(model, ts_compile, partition_fn=partition_with_recompute_fwd_in_bwd, decompositions=decomposition_table)

    batches = input_func(args.warmup_steps+args.steps, args.input_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.input_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    with torch.autograd.profiler.emit_nvtx(enabled=args.profile_with_nvtx):
        with torch.jit.fuser('fuser2') :
            for step,batch in enumerate(batches) :
                if step == args.warmup_steps :
                    torch.cuda.profiler.start()
                    start_evt.record()
 
                with torch.cuda.amp.autocast(enabled=args.amp) :
                    loss = aot_model(*batch)
                if grads :
                    scaler.scale(loss).backward(grads[step])
                else :
                    scaler.scale(loss).backward()
  
                if step % args.grad_accum_steps == 0 :
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
    
    stop_evt.record()
    stop_evt.synchronize()
    return start_evt.elapsed_time(stop_evt)
