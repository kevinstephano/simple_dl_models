import argparse
import os
import random
import torch
import pandas as pd

pd.options.display.max_colwidth=100

from execution import execution_loop

def run(sys_argv, model_name, model, optim_func, input_func, grad_func) : 
    parser = argparse.ArgumentParser(description='DL Models Runner')
    parser.add_argument('--warmup_steps', default=20, type=int, help='Warmup model steps.')
    parser.add_argument('--steps', default=50, type=int, help='Model steps.')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='Steps per optimizer step')
    parser.add_argument('--seed', default=42, type=int, help='Random Seed.')
    parser.add_argument('--model_dtype', default='torch.float32', type=str, help='Model data type.')
    parser.add_argument('--input_dtype', default='torch.float32', type=str, help='Input data type.')
    parser.add_argument('--amp', default=False, action='store_true', help='Run with AMP autocast and GradScaler when using FP16 inputs.')
    parser.add_argument('--fp16', default=False, action='store_true', help='Run with only GradScaler with FP16 inputs and FP16 model parameters.')
    parser.add_argument('--grad_scaler', default=False, action='store_true', help='Run with GradScaler when using FP16 inputs.')
    parser.add_argument('--device', default='cuda', type=str, help='Device type.')
    parser.add_argument('--jit_script', default=False, action='store_true', help='Run with jit.script model.')
    parser.add_argument('--aot_autograd', default=False, action='store_true', help='Run with AOT Autograd.')
    parser.add_argument('--torchdynamo', default=False, action='store_true', help='Run with Torch Dynamo.')
    parser.add_argument('--ltc', default=False, action='store_true', help='Run with Lazy Tensors.')
    parser.add_argument('--profile_with_nvtx', default=False, action='store_true', help='Enable NVTX markers when profiling.')
    parser.add_argument('--skip_eager', default=False, action='store_true', help='Skip the Eager Mode comparison.')
    parser.add_argument('--inference', default=False, action='store_true', help='Run inference.')

    args,extra_args = parser.parse_known_args(args=sys_argv[1:])

    assert len(extra_args) == 0, "Unknown args: {}".format(extra_args)

    #########################################################
    # EDIT ARGUMENTS: for data types and AMP/GradScaler usage
    args.model_dtype = eval(args.model_dtype)
    args.input_dtype = eval(args.input_dtype)
    if args.amp :
        assert args.model_dtype == torch.float32, "Model dtype is incorrect for amp usage: {}".format(args.model_dtype)
        args.input_dtype = torch.float16
        args.grad_scaler = True
    if args.grad_scaler :
        assert args.input_dtype == torch.float16, "Input dtype is incorrect for GradScaler usage: {}".format(args.input_dtype)
    if args.fp16 :
        args.input_dtype = torch.float16
        args.model_dtype = torch.float16
        args.grad_scaler = True

    #########################################################

    tests = []
    if args.jit_script :
        tests.append("jit_script")
    if args.aot_autograd :
        tests.append("aot_autograd")
    if args.torchdynamo :
        tests.append("torchdynamo")

    result_records = []
   
    # Run eager mode first
    eager_record = None
    if not args.skip_eager :
        torch.cuda.empty_cache()
        random.seed(a=args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.random.manual_seed(args.seed)
        eager_record = execution_loop.execute(args, "eager", model_name, model, optim_func, input_func, grad_func, None)
        result_records.append(eager_record)

    # Run specified engines
    test_times = []
    for name in tests :
        torch.cuda.empty_cache()
        random.seed(a=args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.random.manual_seed(args.seed)
        result_records.append(execution_loop.execute(args, name, model_name, model, optim_func, input_func, grad_func, eager_record))

    print(pd.DataFrame(result_records))
    return result_records
