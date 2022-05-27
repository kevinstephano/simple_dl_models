import torch
import importlib
import pip

try:
    importlib.import_module('torchdynamo')
    print('Successfully imported torchdynamo!')
except ModuleNotFoundError:
    print("Downloading torchdynamo...")
    pip.main(['install', '-e', 'git+https://github.com/pytorch/torchdynamo.git#egg=torchdynamo'])
finally:
    importlib.import_module('torchdynamo')

try:
    importlib.import_module('functorch')
    print('Successfully imported functorch!')
except ModuleNotFoundError:
    print("Downloading functorch...")
    pip.main(['install', '-e', 'git+https://github.com/pytorch/functorch.git#egg=functorch'])
finally:
    importlib.import_module('functorch')

# A functorch dependency for AOT Autograd
try:
    importlib.import_module('networkx')
    print('Successfully imported networkx!')
except ModuleNotFoundError:
    print("Downloading functorch...")
    pip.main(['install', 'networkx'])
finally:
    importlib.import_module('networkx')

import torchdynamo
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy

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
            with torchdynamo.optimize(aot_autograd_speedup_strategy):
                for step,batch in enumerate(batches) :
                    if step == args.warmup_steps :
                        torch.cuda.profiler.start()
                        start_evt.record()
           
                    if not args.inference :
                        with torch.cuda.amp.autocast(enabled=args.amp):
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
                            with torch.cuda.amp.autocast(enabled=args.amp):
                                loss = model(*batch)
   
    stop_evt.record()
    stop_evt.synchronize()
    return start_evt.elapsed_time(stop_evt)
