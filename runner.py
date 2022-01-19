import argparse
import torch

import eager_engine

def run(sys_argv, model, optim_func, data_func, grad_func) : 
    parser = argparse.ArgumentParser(description='DL Models Runner')
    parser.add_argument('--warmup_steps', default=5, type=int, help='Model steps.')
    parser.add_argument('--steps', default=20, type=int, help='Model steps.')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='Steps per optimizer step')
    parser.add_argument('--model_dtype', default=torch.float32, help='Model data type.')
    parser.add_argument('--data_dtype', default=torch.float32, help='Input data type.')
    parser.add_argument('--device', default='cuda', type=str, help='Device type.')
    parser.add_argument('--jit_script', default=False, action='store_true', help='Run with jit.script model.')
    parser.add_argument('--aot_autograd', default=False, action='store_true', help='Run with AOT Autograd.')
    parser.add_argument('--ltc', default=False, action='store_true', help='Run with Lazy Tensors.')

    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    tests = []
    if args.jit_script :
        import jit_script_engine
        tests.append(("JIT_Script", jit_script_engine))
    if args.aot_autograd :
        import aot_autograd_engine
        tests.append(("AOT_Autograd", aot_autograd_engine))
    if args.ltc :
        import ltc_engine
        tests.append(("LTC", ltc_engine))

    test_times = []
    for name,engine in tests :
        test_times.append((name, engine.train_loop(args, model, optim_func, data_func, grad_func) / args. steps * 1000.0))
        
    eager_time = eager_engine.train_loop(args, model, optim_func, data_func, grad_func) / args.steps * 1000.0

    timing_str = ">>> Eager-Time(us): {:.3f}".format(eager_time)

    for name,test_time in test_times :
        timing_str += " {}-Time(us): {:.3f} {}-Speedup: {:.2f}".format(name, test_time, name, eager_time/test_time)

    print(timing_str)
