import argparse
import torch

from engines import eager_engine

def run(sys_argv, model, optim_func, input_func, grad_func) : 
    parser = argparse.ArgumentParser(description='DL Models Runner')
    parser.add_argument('--warmup_steps', default=5, type=int, help='Model steps.')
    parser.add_argument('--steps', default=20, type=int, help='Model steps.')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='Steps per optimizer step')
    parser.add_argument('--model_dtype', default='torch.float32', type=str, help='Model data type.')
    parser.add_argument('--input_dtype', default='torch.float32', type=str, help='Input data type.')
    parser.add_argument('--amp', default=False, action='store_true', help='Run with AMP when using FP16 inputs.')
    parser.add_argument('--grad_scaler', default=False, action='store_true', help='Run with GradScaler when using FP16 inputs.')
    parser.add_argument('--device', default='cuda', type=str, help='Device type.')
    parser.add_argument('--jit_script', default=False, action='store_true', help='Run with jit.script model.')
    parser.add_argument('--aot_autograd', default=False, action='store_true', help='Run with AOT Autograd.')
    parser.add_argument('--ltc', default=False, action='store_true', help='Run with Lazy Tensors.')

    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    # Edit Arguments for data types and AMP/GradScaler usage
    args.model_dtype = eval(args.model_dtype)
    args.input_dtype = eval(args.input_dtype)
    if args.amp :
        assert args.model_dtype == torch.float32, "Model dtype is incorrect for amp usage: {}".format(args.model_dtype)
        args.input_dtype = torch.float16
        args.grad_scaler = True
    if args.grad_scaler :
        assert args.input_dtype == torch.float16, "Input dtype is incorrect for GradScaler usage: {}".format(args.input_dtype)

    tests = []
    if args.jit_script :
        from engines import jit_script_engine
        tests.append(("JIT_Script", jit_script_engine))
    if args.aot_autograd :
        from engines import aot_autograd_engine
        tests.append(("AOT_Autograd", aot_autograd_engine))
    if args.ltc :
        from engines import ltc_engine
        tests.append(("LTC", ltc_engine))

    test_times = []
    for name,engine in tests :
        test_times.append((name, engine.train_loop(args, model, optim_func, input_func, grad_func) / args. steps * 1000.0))
        
    eager_time = eager_engine.train_loop(args, model, optim_func, input_func, grad_func) / args.steps * 1000.0

    timing_str = ">>> Eager-Time(us): {:.3f}".format(eager_time)

    for name,test_time in test_times :
        timing_str += " {}-Time(us): {:.3f} {}-Speedup: {:.2f}".format(name, test_time, name, eager_time/test_time)

    print(timing_str)
