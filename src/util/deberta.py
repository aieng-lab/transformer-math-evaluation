import json
from typing import Mapping, Any

import torch
from torch import nn
from transformers import AutoModel, DebertaV2Tokenizer, DebertaV2Model, AutoConfig, AutoTokenizer
import pathlib

class DebertaWithPoolingLayeOldr(nn.Module):
    def __init__(self, pretrained_model_name):
        super(DebertaWithPoolingLayer, self).__init__()

        # Load the Deberta model and tokenizer
        self.deberta = DebertaV2Model.from_pretrained(pretrained_model_name)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name)

        # Add a pooling layer (Linear + tanh activation) for the CLS token
        self.pooling_layer = nn.Sequential(
            nn.Linear(self.deberta.config.hidden_size, self.deberta.config.hidden_size),
            nn.Tanh()
        )
        try:
            state_dict = torch.load(pretrained_model_name + '/pooling.bin')
            self.pooling_layer[0].load_state_dict(state_dict)
        except FileNotFoundError:
            print("Initialize new DeBERTa Pooling Layer with random values")

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # Forward pass through the Deberta model
        outputs = self.deberta(input_ids, attention_mask=attention_mask, *args, **kwargs)

        # Extract the hidden states from the output
        hidden_states = outputs.last_hidden_state

        # Get the CLS token representation (first token)
        cls_token = hidden_states[:, 0, :]

        # Apply the pooling layer to the CLS token representation
        pooled_output = self.pooling_layer(cls_token)

        # Include the pooled_output in the output dictionary as 'pooling_layer'
        outputs["pooling_layer"] = pooled_output

        return outputs

    def save_pretrained(self, path, include_pooler=True):
        self.deberta.save_pretrained(path)
        if include_pooler:
            torch.save(self.pooling_layer[0].state_dict(), path + '/pooling.bin')

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        deberta_state_dict = {k.removeprefix('deberta.'): v for k, v in state_dict.items() if k.startswith('deberta')}
        pooler_state_dict = {k.removeprefix('pooling_layer.0.'): v for k, v in state_dict.items() if k.startswith('pooling')}
        self.deberta.load_state_dict(deberta_state_dict, strict=strict)
        self.pooling_layer[0].load_state_dict(pooler_state_dict)

class DebertaWithPoolingLayer(nn.Module):
    def __init__(self, pretrained_model_name):
        super(DebertaWithPoolingLayer, self).__init__()

        # Load the Deberta model and tokenizer
        self.deberta = DebertaV2Model.from_pretrained(pretrained_model_name)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name)

        # Add a pooling layer (Linear + tanh activation) for the CLS token
        self.pooling_layer = nn.Sequential(
            nn.Linear(self.deberta.config.hidden_size, self.deberta.config.hidden_size),
            nn.Tanh()
        )

        self.config = self.deberta.config
        self.embeddings = self.deberta.embeddings

        try:
            state_dict = torch.load(pretrained_model_name + '/pooling.bin')
            self.pooling_layer[0].load_state_dict(state_dict)
        except FileNotFoundError:
            pass

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # Forward pass through the Deberta model
        outputs = self.deberta(input_ids, attention_mask=attention_mask, *args, **kwargs)

        # Extract the hidden states from the output
        hidden_states = outputs.last_hidden_state

        # Get the CLS token representation (first token)
        cls_token = hidden_states[:, 0, :]

        # Apply the pooling layer to the CLS token representation
        pooled_output = self.pooling_layer(cls_token)
        # Include the pooled_output in the output dictionary as 'pooling_layer'
        outputs["pooler_output"] = pooled_output

        return outputs

    def save_pretrained(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict, configuration, and tokenizer
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        self.deberta.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        deberta_state_dict = {k.removeprefix('deberta.'): v for k, v in state_dict.items() if k.startswith('deberta')}
        pooler_state_dict = {k.removeprefix('pooling_layer.0.'): v for k, v in state_dict.items() if k.startswith('pooling')}
        self.deberta.load_state_dict(deberta_state_dict, strict=strict)
        self.pooling_layer[0].load_state_dict(pooler_state_dict)

    @classmethod
    def from_pretrained(cls, name):
        # Initialize the instance
        instance = cls(name)

        try:
            # Load the model's state_dict
            instance.load_state_dict(torch.load(f"{name}/pytorch_model.bin"))

            # Load the configuration and tokenizer
            instance.deberta.config = AutoConfig.from_pretrained(name)
            instance.tokenizer = AutoTokenizer.from_pretrained(name)
        except FileNotFoundError:
            print("Could not find DeBERTa pooling layer. Initialize new values")

        return instance
