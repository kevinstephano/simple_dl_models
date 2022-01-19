# Models and Parts of Models to Test the Pytorch Fusion Frontends

## Usage

```
python [model file] [--jit_script] [--ltc] [--aot_autograd] 
```

If you include no options with the model file, only eager mode will be timed.

## Example Output

```
$ python simple_model.py --jit_script
>>> Eager-Time(us): 411.493 JIT_Script-Time(us): 368.355 JIT_Script-Speedup: 1.12
```
