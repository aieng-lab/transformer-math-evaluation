from typing import Any, Mapping

import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, Trainer, AutoConfig, AutoTokenizer
import pathlib

from util import create_model


class BertForInformationRetrieval(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForInformationRetrieval, self).__init__()
        model = create_model
        self.bert = model(bert_model_name)
        num_labels = 2
        self.ir_classification_layer = nn.Linear(self.bert.config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.ir_classification_layer.weight)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=kwargs.get('token_type_ids'))
        cls_output = outputs.pooler_output
        logits = self.ir_classification_layer(cls_output)
        probability = torch.sigmoid(logits.squeeze(-1))
        return probability

    def save_pretrained(self, output_directory):
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict, configuration, and tokenizer
        torch.save(self.state_dict(), f"{output_directory}/pytorch_model.bin")
        self.bert.config.save_pretrained(output_directory)
        if hasattr(self.bert, 'tokenizer'):
            self.bert.tokenizer.save_pretrained(output_directory)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if 'deberta' in str(type(self.bert)):
            deberta_state_dict = {k.removeprefix('bert.deberta.'): v for k, v in state_dict.items() if k.startswith('bert.deberta')}
            pooler_state_dict = {k.removeprefix('bert.pooling_layer.0.'): v for k, v in state_dict.items() if k.startswith('bert.pooling')}
            ir_state_dict = {k.removeprefix('ir_classification_layer.'): v for k, v in state_dict.items() if not k.startswith('bert')}

            self.bert.deberta.load_state_dict(deberta_state_dict, strict=strict)
            self.bert.pooling_layer[0].load_state_dict(pooler_state_dict)
            self.ir_classification_layer.load_state_dict(ir_state_dict)
        else:
            super().load_state_dict(state_dict, strict)

    @classmethod
    def from_pretrained(cls, name):
        # Initialize the instance
        instance = cls(name)

        try:
            # Load the model's state_dict
            state_dict = torch.load(f"{name}/pytorch_model.bin")
            if any(k.startswith('bert') for k in state_dict):
                instance.load_state_dict(state_dict)
            else:
                instance.bert.load_state_dict(state_dict)

            # Load the configuration and tokenizer
            instance.bert.config = AutoConfig.from_pretrained(name)
        except Exception as e:
            print("Could not find IR Linear layer. Initialize new values: %s" % e)

        return instance

class BertForInformationRetrievalV2(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForInformationRetrievalV2, self).__init__()
        self.bert = create_model(bert_model_name)
        num_labels = 1
        self.ir_classification_layer = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # token_type_ids are not used by witiko/mathberta
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=kwargs.get('token_type_ids'))
        cls_output = outputs.pooler_output
        #cls_output = outputs.last_hidden_state[:, 0, :] #+ todo remove
        logits = self.ir_classification_layer(cls_output)
        probability = torch.sigmoid(logits.squeeze(-1))
        return probability

    def save_pretrained(self, output_directory):
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict, configuration, and tokenizer
        torch.save(self.state_dict(), f"{output_directory}/pytorch_model.bin")
        self.bert.config.save_pretrained(output_directory)

    @classmethod
    def from_pretrained(cls, name):
        # Initialize the instance
        instance = cls(name)

        try:
            # Load the model's state_dict
            instance.load_state_dict(torch.load(f"{name}/pytorch_model.bin"))

            # Load the configuration and tokenizer
            instance.bert.config = AutoConfig.from_pretrained(name)
        except FileNotFoundError:
            print("Could not find IR Linear layer. Initialize new values")

        return instance


if __name__ == '__main__':
    model = 'bert-base-cased'
    #model = 'microsoft/deberta-v3-base'
    bert_ir = BertForInformationRetrieval.from_pretrained(model)
    bias = torch.Tensor([42.0, 73.0])
    bias = nn.Parameter(bias)
    bert_ir.ir_classification_layer.bias = bias

    bert_ir.save_pretrained('test')

    bert_ir2 = BertForInformationRetrieval.from_pretrained('test')
    assert all(bert_ir2.ir_classification_layer.bias == bias)
