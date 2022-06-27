import math
import importlib
import pandas as pd
import torch
import subprocess
import sys

pd.options.display.max_colwidth=100

from execution import runner
from components.apex_adam_optimizer import optim_func

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    importlib.import_module('transformers')
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install('git+https://github.com/huggingface/transformers.git#egg=transformers')
finally:
    from transformers import T5Config, T5ForConditionalGeneration


def t5_input_func(steps, dtype, device) :
    vocab_size = 32128
    sequences = 8
    src_sequence_length = 512
    tgt_sequence_length = 128
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, src_sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, src_sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, tgt_sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, None, None, None, None, None, labels, None, None, None, False])
    return results

if __name__ == "__main__" :

    final_results = []

    config = T5Config.from_pretrained('t5-large')
    final_results += runner.run(sys.argv, 'T5ForConditionalGeneration_t5-large_[seqs=8,src_seql=512,tgt_seql=128]', T5ForConditionalGeneration(config), optim_func, t5_input_func, None)

    print('=========================== Final Results ===========================')
    print(pd.DataFrame(final_results))
