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
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers import RobertaConfig, RobertaForMaskedLM
    from transformers import AlbertConfig, AlbertForPreTraining
    from transformers import T5Config, T5ForConditionalGeneration
    from transformers import DebertaConfig, DebertaForMaskedLM
    from transformers import XLNetConfig, XLNetLMHeadModel

def bert_p1_input_func(steps, dtype, device) :
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

def roberta_input_func(steps, dtype, device) :
    vocab_size = 50265
    sequences = 64
    sequence_length = 128
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, None, None, labels, None, None, False])
    return results

def albert_input_func(steps, dtype, device) :
    vocab_size = 30000
    sequences = 8
    sequence_length = 512
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        sentence_order_label = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, labels, sentence_order_label, None, None, False])
    return results

def t5_input_func(steps, dtype, device) :
    vocab_size = 32128
    sequences = 4
    sequence_length = 512
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, None, None, None, None, None, labels, None, None, None, False])
    return results

def deberta_input_func(steps, dtype, device) :
    vocab_size = 50265
    sequences = 8
    sequence_length = 512
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, labels, None, None, False])
    return results

def xlnet_input_func(steps, dtype, device) :
    vocab_size = 32000
    sequences = 16
    sequence_length = 512
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        attention_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        labels = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, attention_mask, None, None, None, None, None, None, None, labels, None, None, False])
    return results

if __name__ == "__main__" :

    final_results = []

    config = BertConfig.from_pretrained('bert-large-uncased')
    final_results += runner.run(sys.argv, 'BertForPreTraining_P1_bert-large-uncased_[seqs=64,seql=128]', BertForPreTraining(config), optim_func, bert_p1_input_func, None)
    final_results += runner.run(sys.argv, 'BertForPreTraining_P2_bert-large-uncased_[seqs=16,seql=512]', BertForPreTraining(config), optim_func, bert_p2_input_func, None)

    config = GPT2Config.from_pretrained('gpt2-large')
    final_results += runner.run(sys.argv, 'GPT2LMHeadModel_gpt2-large_[seqs=2,seql=1024]', GPT2LMHeadModel(config), optim_func, gpt2_input_func, None)

    config = RobertaConfig.from_pretrained('roberta-large')
    final_results += runner.run(sys.argv, 'RobertaForMaskedLM_roberta-large_[seqs=64,seql=128]', RobertaForMaskedLM(config), optim_func, roberta_input_func, None)

    config = AlbertConfig.from_pretrained('albert-xxlarge-v2')
    final_results += runner.run(sys.argv, 'AlbertForPreTraining_albert-xxlarge-v2_[seqs=8,seql=512]', AlbertForPreTraining(config), optim_func, albert_input_func, None)

    config = T5Config.from_pretrained('t5-large')
    final_results += runner.run(sys.argv, 'T5ForConditionalGeneration_t5-large_[seqs=4,seql=512]', T5ForConditionalGeneration(config), optim_func, t5_input_func, None)

    config = DebertaConfig.from_pretrained('microsoft/deberta-large')
    final_results += runner.run(sys.argv, 'DebertaForMaskedLM_deberata-large_[seqs=8,seql=512]', DebertaForMaskedLM(config), optim_func, deberta_input_func, None)

    config = XLNetConfig.from_pretrained('xlnet-large-cased')
    final_results += runner.run(sys.argv, 'XLNetLMHeadModel_xlnet-large-cased_[seqs=16,seql=512]', XLNetLMHeadModel(config), optim_func, xlnet_input_func, None)

    print('=========================== Final Results ===========================')
    print(pd.DataFrame(final_results))
