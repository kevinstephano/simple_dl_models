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
    from transformers import BertConfig, BertForPreTraining

def bert_p2_input_func(steps, dtype, device) :
    vocab_size = 30522
    sequences = 16
    sequence_length = 512
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        next_sentence_labels = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, labels, next_sentence_labels, None, None, False])
    return results

if __name__ == "__main__" :

    final_results = []

    config = BertConfig.from_pretrained('bert-large-uncased')
    final_results += runner.run(sys.argv, 'BertForPreTraining_P2_bert-large-uncased_[seqs=16,seql=512]', BertForPreTraining(config), optim_func, bert_p2_input_func, None)

    print('=========================== Final Results ===========================')
    print(pd.DataFrame(final_results))
