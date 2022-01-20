# Models and Model Blocks to Test the Pytorch Fusion Frontends

## Basic Usage

```
python [model file] [engine: --jit_script|--ltc|--aot_autograd] 
```
If you include no options with the model file, only eager mode will be timed

## Profiling Usage
There are two profiling scripts:
* `profile_all.sh` : profiles everything that runs
* `profile_api_start.sh` : Only profiles after the Cuda API is called to start profiling after warmup

```
./scripts/profile_api_start.sh python [model file] [engine: --jit_script|--ltc|--aot_autograd] [--profile_with_nvtx]
```

## Example Output

```
$ python simple_model.py --jit_script
>>> Eager-Time(us): 411.493 JIT_Script-Time(us): 368.355 JIT_Script-Speedup: 1.12
```

## Mixed Precision Usage
_Defaults to FP32 model and input data._
### AMP Usage
Model parameters remain in FP32 and input data is in FP16
```
python [model file] [engine: --jit_script|--ltc|--aot_autograd] --amp
```
### Mixed Precision with Model in FP16 and GradScaler (Advanced Performace Usage)
Model parameters are in FP16 and input data is in FP16
```
python [model file] [engine: --jit_script|--ltc|--aot_autograd] --max_fp16_perf

or

python [model file] [engine: --jit_script|--ltc|--aot_autograd] --grad_scaler --input_dtype=torch.float16 --model_dtype=torch.float16
```
_This set of options does not work with Optimizers that rely on GradScaler to do the unscaling (Native Pytorch Optimizers) as it asserts on FP16 weights. For test purposes, just don't use the `--grad_scaler` flag._

## Options
### Engines

You have your choice of 4 different front engines.  The Eager Engine is always run as a comparison point for speedup.  If you don't specify an engine, just the Eager Engine will be run.  Besides the Eager Engine, the rest give opportunities for fusion to the NVFuser backend for GPUs.

Engines:
* **Eager**: _Default: no switch needed._
* **JIT Script**: `--jit_script`
* **Lazy Tensor Core**: `--ltc`
* **AOT Autograd**: `--aot_autograd`

## Single GPU Models
### Simple
* Simple linear layer + relu and SGD Optimizer: `simple_model.py`
### Transformer Model Components
* Multihead Attention Block with no optimizer: `xformer_multihead_attn.py`
* Feed Forward Block with no optimizer: `xformer_feed_fwd.py`
* One Encoder Layer with no optimizer: `xformer_1_layer.py`
### Bert Models
#### Fixed Batch Size Models
* Full Bert Model (bert-large) with APEX Lamb Optimizer: `bert_model.py`
* Full Bert Model (bert-large) with Native AdamW Optimizer: `bert_model_adam_opt.py`
* Bert Model with 1 Layer (bert-large sized) with no optimizer: `bert_model_1_layer_no_opt.py`
#### Dynamic Batch Size Models (sequence length per batch varies)
* Full Bert Model (bert-large) with APEX Lamb Optimizer: `dynamic_bert_model.py`
* Full Bert Model (bert-large) with Native AdamW Optimizer: `dynamic_bert_model_adam_opt.py`
* Bert Model with 1 Layer (bert-large sized) with no optimizer: `dynamic_bert_model_1_layer_no_opt.py`
