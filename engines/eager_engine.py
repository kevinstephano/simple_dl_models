import torch

def train_loop(args, model, optim_func, data_func, grad_func=None) :
    model.to(device=args.device)
   
    optimizer = optim_func(model.parameters())

    batches = data_func(args.warmup_steps+args.steps, args.data_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.data_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    for step,batch in enumerate(batches) :
        if step == args.warmup_steps :
            start_evt.record()

        loss = model(*batch)
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
