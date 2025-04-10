from transformers import AutoModel
import json

from util.deberta import DebertaWithPoolingLayer


def create_model(model_identifier):
    if 'deberta' in model_identifier:
        try:
            return DebertaWithPoolingLayer(model_identifier)
        except Exception:
            pass

    try:
        config = json.load(open(model_identifier + '/config.json', 'r+'))
        architectures = config.get('architectures', [])
        model_type = config.get('model_type', '')
        if any('deberta' in s.lower() for s in architectures) or 'deberta' in model_type.lower():
            return DebertaWithPoolingLayer(model_identifier)
    except FileNotFoundError:
        pass

    model = AutoModel.from_pretrained(model_identifier)
    return model