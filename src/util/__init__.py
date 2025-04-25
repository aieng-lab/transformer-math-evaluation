from transformers import AutoModel

def create_model(model_identifier):
    return AutoModel.from_pretrained(model_identifier)

def remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string