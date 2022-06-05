import math
import importlib
import torch
import subprocess
import sys

from execution import runner
from torch.optim import AdamW

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    importlib.import_module('transformers')
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install('git+https://github.com/huggingface/transformers.git#egg=transformers')
finally:
    from transformers import BertConfig, BertForPreTraining
    from transformers import GPT2Config, GPT2LMHeadModel

def bert_input_func(steps, dtype, device) :
    vocab_size = 30522
    sequences = 64
    sequence_length = 128
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        next_sentence_labels = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, labels, next_sentence_labels, None, None, False])
    return results

def gpt2_input_func(steps, dtype, device) :
    vocab_size = 50257
    sequences = 2
    sequence_length = 1024
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, None, attention_mask, None, None, None, None, None, None, labels, None, None, None, False])
    return results

def optim_func(params) :
    return AdamW(params)

if __name__ == "__main__" :
    config = BertConfig.from_pretrained('bert-large-uncased')
    runner.run(sys.argv, 'BertForPreTraining_bert-large-uncased_[seqs=64,seql=128]', BertForPreTraining(config), optim_func, bert_input_func, None) 
    config = GPT2Config.from_pretrained('gpt2-large')
    runner.run(sys.argv, 'GPT2LMHeadModel_gpt2-large_[seqs=2,seql=1024]', GPT2LMHeadModel(config), optim_func, gpt2_input_func, None) 
