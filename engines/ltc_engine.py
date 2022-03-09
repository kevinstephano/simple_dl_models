import torch

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm

# Turning off Horizontal Fusion in NVFuser. A bug is seen with optimizers
torch._C._jit_set_nvfuser_horizontal_mode(False)

def train_loop(args, model, optim_func, input_func, grad_func=None) :
    model.to(device=args.device)
    model.to(dtype=args.model_dtype)
   
    optimizer = optim_func(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=(args.grad_scaler and not hasattr(optimizer, '_dummy')))

    batches = input_func(args.warmup_steps+args.steps, args.input_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.input_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    with torch.autograd.profiler.emit_nvtx(enabled=args.profile_with_nvtx):
        with torch.jit.fuser('fuser2') :
            for step,batch in enumerate(batches) :
                if not args.inference :
                    with torch.cuda.amp.autocast(enabled=args.amp) :
                        loss = model(*batch)
                    if grads :
                        scaler.scale(loss).backward(grads[step])
                    else :
                        scaler.scale(loss).backward()
                 
                    if step % args.grad_accum_steps == 0 :
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else :
                    with torch.inference_mode() :
                        with torch.cuda.amp.autocast(enabled=args.amp) :
                            loss = model(*batch)
 
                ltm.mark_step()
 
                if step == args.warmup_steps :
                    torch.cuda.profiler.start()
                    start_evt.record()
    
    ltm.wait_device_ops()
    stop_evt.record()
    stop_evt.synchronize()
    return start_evt.elapsed_time(stop_evt)
