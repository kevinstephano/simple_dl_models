import argparse
import torch

import jit_script_engine

def run(sys_argv, model, optim_func, data_func) : 
    parser = argparse.ArgumentParser(description='DL Models Runner')
    parser.add_argument('--steps', default=15, type=int, help='Model steps.')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='Steps per optimizer step')
    parser.add_argument('--model_dtype', default=torch.float32, help='Model data type.')
    parser.add_argument('--data_dtype', default=torch.float32, help='Input data type.')
    parser.add_argument('--device', default='cuda', type=str, help='Device type.')
    parser.add_argument('--jit_script', default=False, action='store_true', help='Run with jit.script model.')
    parser.add_argument('--aot_autograd', default=False, action='store_true', help='Run with AOT Autograd.')
    parser.add_argument('--ltc', default=False, action='store_true', help='Run with Lazy Tensors.')

    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    if args.jit_script :
        jit_script_engine.train_loop(args, model, optim_func, data_func)
    else :
        assert False, "No specified model engine."

