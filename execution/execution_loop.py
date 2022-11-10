import gc
import importlib
import subprocess
import sys
import torch
import torch._dynamo
torch.backends.cuda.matmul.allow_tf32 = True

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def result_record(args, exec_name, model_name, exec_time, gpu_memory, compare_record):
    return {
         "model_name": model_name,
         "executor": exec_name,
         "run_type": ('inference' if args.inference else 'training'),
         "data_dtype": args.input_dtype,
         "model_dtype": args.model_dtype,
         "time(us)": exec_time,
         "memory(GB)": gpu_memory, 
         "speedup": (round(compare_record['time(us)'] / exec_time, 2) if compare_record is not None else 1.0),
         "memory_compression": (round(compare_record['memory(GB)'] / gpu_memory, 2) if compare_record is not None else 1.0),
    }

def execute(args, exec_name, model_name, model, optim_func, input_func, grad_func=None, compare_record=None) :
    if 'GNN' in model_name:
        try:
            import torch_geometric
        except:
            print('Installing PyG...')
            pip_install('torch-scatter')
            pip_install('torch-sparse')
            pip_install('torch-cluster')
            pip_install('torch-spline-conv')
            pip_install('torch-geometric')
        def forward_pass(model, batch):
            if isinstance(batch, torch_geometric.data.Data):
                return model(batch.x, batch.edge_index, batch.y)
            elif isinstance(batch, torch_geometric.data.HeteroData):
                return model(batch.collect('x'), batch.collect('edge_index'), batch[batch.labeled_node_type].y)
    else:
        def forward_pass(model, batch):
            return model(*batch)

    optimize_func = None
    if exec_name == 'jit_script':
        torch._C._jit_set_autocast_mode(True)
    else :
        torch._C._jit_set_autocast_mode(False)
    if exec_name == 'torchdynamo' or exec_name == 'aot_autograd':
        try:
            importlib.import_module('functorch')
        except ModuleNotFoundError:
            print("Installing functorch...")
            pip_install('git+https://github.com/pytorch/functorch.git#egg=functorch[aot]')
        finally:
            if exec_name == 'aot_autograd':
                from functorch.compile import memory_efficient_fusion
    if exec_name == 'inductor' or exec_name == 'nvprims_nvfuser':
        if 'GNN' in model_name:
            torch._dynamo.config.fake_tensor_propagation=False
        optimize_func = torch._dynamo.optimize(exec_name)

        @torch._dynamo.skip
        def get_cur_memory():
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.current"]
            return peak_bytes_requirement / 1.e9
    else :
        def get_cur_memory():
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.current"]
            return peak_bytes_requirement / 1.e9

    model.to(device=args.device)
    model.to(dtype=args.model_dtype)

    optimizer = optim_func(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=(args.grad_scaler and not hasattr(optimizer, '_dummy')))

    if exec_name == 'aot_autograd':
        model = memory_efficient_fusion(model)
    elif exec_name == 'jit_script':
        model = torch.jit.script(model)
    elif exec_name == 'inductor':
        model = optimize_func(model)
    elif exec_name == 'nvprims_nvfuser':
        model = optimize_func(model)

    batches = input_func(args.warmup_steps+args.steps, args.input_dtype, args.device)
    grads = None
    if grad_func :
        grads = grad_func(args.warmup_steps+args.steps, args.input_dtype, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    gpu_memory = 0.0
    with torch.autograd.profiler.emit_nvtx(enabled=args.profile_with_nvtx):
        for step,batch in enumerate(batches) :
            if step == args.warmup_steps :
                torch.cuda.profiler.start()
                start_evt.record()

            if not args.inference :
                with torch.cuda.amp.autocast(enabled=args.amp):
                    loss = forward_pass(model, batch)
                    if args.warmup_steps > 4 and step == 4:
                        gpu_memory = get_cur_memory()
                    if grads :
                        scaler.scale(loss[0]).backward(grads[step])
                    else :
                        scaler.scale(loss[0]).backward()

                if step % args.grad_accum_steps == 0 :
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else :
                with torch.inference_mode(), torch.jit.fuser('fuser2'), optimize_ctx:
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        loss = forward_pass(model, batch)
                    if args.warmup_steps > 4 and step == 4:
                        gpu_memory = get_cur_memory() 
   
    stop_evt.record()
    start_evt.synchronize()
    stop_evt.synchronize()
    exec_time = round(start_evt.elapsed_time(stop_evt) / args.steps * 1000.0, 3)

    return result_record(args, exec_name, model_name, exec_time, gpu_memory, compare_record)
