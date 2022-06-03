import torch
import importlib
import pip
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def result_record(args, exec_name, model, exec_time, gpu_memory, compare_record):
    return {
         "model_name": model.name,
         "executor": exec_name,
         "data_dtype": args.input_dtype,
         "model_dtype": args.model_dtype,
         "time(us)": exec_time,
         "memory(GB)": gpu_memory,
         "speedup": (round(compare_record['time(us)'] / exec_time, 2) if compare_record is not None else 1.0)
         "memory_compression": (round(compare_record['memory_compression'] / gpu_memory, 2) if compare_record is not None else 1.0)
    }

def execute(args, exec_name, model, optim_func, input_func, compare_record=None, grad_func=None) :
    optimize_ctx = NullContext
    if args.torchdynamo or args.aot_autograd:
        try:
            importlib.import_module('functorch')
            print('Successfully imported functorch!')
        except ModuleNotFoundError:
            print("Installing functorch...")
            pip_install('git+https://github.com/pytorch/functorch.git#egg=functorch[aot]')
        finally:
            if args.aot_autograd:
                from functorch.compile import memory_efficient_fusion
    if args.torchdynamo:
        try:
            importlib.import_module('torchdynamo')
            print('Successfully imported torchdynamo!')
        except ModuleNotFoundError:
            print("Installing torchdynamo...")
            pip_install('git+https://github.com/pytorch/torchdynamo.git#egg=torchdynamo')
        finally:
            import torchdynamo
            from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
            optimize_ctx = torchdynamo.optimize(aot_autograd_speedup_strategy)

    model.to(device=args.device)
    model.to(dtype=args.model_dtype)

    optimizer = optim_func(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=(args.grad_scaler and not hasattr(optimizer, '_dummy')))
    
    if args.aot_aotugrad:
        model = memory_efficient_fusion(model)
    elif args.jit_script:
        model = torch.jit.script(model)

    batches = input_func(args.warmup_steps+args.steps, args.input_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.input_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    with torch.autograd.profiler.emit_nvtx(enabled=args.profile_with_nvtx):
        with torch.jit.fuser('fuser2'), optimize_ctx:
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
    exec_time = start_evt.elapsed_time(stop_evt)

    return result_record(args, exec_name, model, exec_time, 1., compare_record)
