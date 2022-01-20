import sys
import math
import torch
from torch import nn

from engines import runner

import xformer_1_layer
import xformer_feed_fwd

from apex.optimizers.fused_mixed_precision_lamb import FusedMixedPrecisionLamb

def optim_func(params) :
    return FusedMixedPrecisionLamb(params)

def input_func(steps, dtype, device) :
    vocab_size = 30528
    sequences = 64
    sequence_length = 128
    results = []
    for _ in range(steps) :
        input_ids = torch.randint(0, vocab_size, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        segment_ids = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        input_mask = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        masked_lm_labels = torch.randint(0, 2, (sequences, sequence_length), device=device, dtype=torch.int64, requires_grad=False)
        next_sentence_labels = torch.randint(0, 2, (sequences,), device=device, dtype=torch.int64, requires_grad=False)
        results.append([input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels])
    return results

class BertConfig :
    def __init__(self) :
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.num_attention_heads = 16
        self.dropout_prob = 0.1
        self.num_hidden_layers = 24
        self.hidden_act = torch.nn.functional.gelu
        self.vocab_size = 30528 # Increase to a multiple of 8
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([xformer_1_layer.BertLayer(config.hidden_size, 
                                                              config.intermediate_size,
                                                              config.num_attention_heads, 
                                                              config.dropout_prob) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.transpose(0,1)
        for i,layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        # The hidden states need to be contiguous at this point to enable
        # dense_sequence_output
        hidden_states = hidden_states.transpose(0,1).contiguous()

        return [hidden_states]

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = xformer_feed_fwd.LinearActivation(config.hidden_size, config.hidden_size, act=torch.tanh)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output

class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoded_layers[-1])

        return encoded_layers, pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense_act = xformer_feed_fwd.LinearActivation(config.hidden_size, config.hidden_size, act=config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_lm_labels):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.criterion = BertPretrainingCriterion(config.vocab_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_labels):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = encoded_layers[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, masked_lm_labels)
        loss = self.criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
        return loss

if __name__ == "__main__" :
    sys.argv.append('--grad_accum_steps=4')
    runner.run(sys.argv, BertForPreTraining(BertConfig()), optim_func, input_func, None) 
