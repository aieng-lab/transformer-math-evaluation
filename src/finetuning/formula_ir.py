import json
import os
from pathlib import Path

import scipy
import random

import sklearn.metrics

import re
import numpy as np
import torch

from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer

from util.bert import BertForInformationRetrieval
from util.training import CustomTrainer, _average_dicts
from util.data import create_data


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, key_label='labels'):
        self.encodings = encodings
        self.labels = labels
        self.idx_ctr = {}
        self.epoch_dependant = isinstance(encodings, dict)
        self.key_label = key_label

        if self.epoch_dependant:
            self.len = len(list(self.labels.values())[0])
        else:
            self.len = len(self.labels)

    def __getitem__(self, idx):
        if self.epoch_dependant:
            if idx in self.idx_ctr:
                self.idx_ctr[idx] += 1
                epoch = self.idx_ctr[idx] % len(self.encodings)
            else:
                epoch = 0
                self.idx_ctr[idx] = 0

            item = {key: val[idx] for key, val in self.encodings[epoch].items()}
            item[self.key_label] = self.labels[epoch][idx]
            return item
        else:
            item = {key: val[idx] for key, val in self.encodings.items()}
            item[self.key_label] = self.labels[idx]
            return item

    def __len__(self):
        return self.len


class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)

        # Calculate the loss
        if len(outputs) == len(labels):
            loss_value = torch.nn.functional.binary_cross_entropy(outputs, labels.type(torch.float))
        else:
            loss_value = torch.nn.functional.cross_entropy(outputs, labels.type(torch.long))

        return (loss_value, outputs) if return_outputs else loss_value

def preprocess_witiko(text):
    # Define the regular expression pattern to match LaTeX formulas
    latex_pattern = r'\$(.*?)\$'

    # Use re.sub() to replace LaTeX formulas with [MATH]...[/MATH]
    preprocessed_text = re.sub(latex_pattern, r'[MATH]\1[/MATH]', text)
    return preprocessed_text

def run_ir(input_model, output_model, data, **kwargs):
    output = output_model.removesuffix('/')
    file_path = output + '/metrics.json'
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.exists(file_path):
        print("File %s already exists, skip this run" % file_path)
        return json.load(open(file_path, 'r', encoding='utf8'))

    key_lhs = kwargs['key_lhs']
    key_rhs = kwargs['key_rhs']
    key_label = 'label'

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_model)

    if kwargs.get('use_text'):
        data_filter = lambda example: example['is_text']
    elif kwargs.get('use_text') is False:
        data_filter = lambda example: not example['is_text']
    else:
        data_filter = None
    batch_size = kwargs.get('batch_size', 16)

    preprocessing = None
    if input_model == 'witiko/mathberta':
        preprocessing = preprocess_witiko

    iteration = kwargs.get('iteration', 0)
    data = create_data(data,
                       max_size=kwargs.get('max_size', None),
                       split_by_formula_name_id=kwargs.get('split_by_formula_name_id', False),
                       epoch_dependent=kwargs.get('epoch_dependent', True),
                       use_challenging_falses=kwargs.get('use_challenging_falses', True),
                       data_filter=data_filter,
                       batch_size=batch_size,
                       preprocessing=preprocessing,
                       seed=iteration,
                       )
    train_dataset = data['train']
    test_dataset = data['test']
    if 'validation' in data:
        val_dataset = data['validation']
    else:
        val_dataset = test_dataset

    # Step 3: Create data loaders
    # Tokenize and encode the input sequences#
    max_length = kwargs.get('max_length', 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use device %s" % device)

    def create_encodings(dataset):
        if isinstance(dataset, dict):
            encodings = {}
            labels = {}
            for i, epoch_dataset in dataset.items():
                epoch_train_encodings = tokenizer(epoch_dataset[key_lhs], epoch_dataset[key_rhs], truncation=True, padding=True, max_length=max_length, return_tensors='pt')
                encodings[i] = epoch_train_encodings

                epoch_train_labels = torch.tensor(epoch_dataset[key_label], dtype=torch.int32, device=device)
                labels[i] = epoch_train_labels
        else:
            encodings = tokenizer(dataset[key_lhs], dataset[key_rhs], truncation=True, padding=True, max_length=max_length, return_tensors='pt')
            labels = torch.tensor(dataset[key_label], dtype=torch.int32, device=device)

        return encodings, labels

    train_encodings, train_labels = create_encodings(train_dataset)
    val_encodings, val_labels = create_encodings(val_dataset)
    test_encodings, test_labels = create_encodings(test_dataset)

    # Step 4: Define the training loop

    # Create dataset objects
    train_dataset = TrainDataset(train_encodings, train_labels)
    val_dataset = TrainDataset(val_encodings, val_labels)
    test_dataset_ = TrainDataset(test_encodings, test_labels)

    model = BertForInformationRetrieval.from_pretrained(input_model).to(device)

    if kwargs.get('bert_freeze', False):
        print("Freeze BERT parameters")
        for _, parameter in model.bert.named_parameters():
            parameter.requires_grad = False

    output_dir = output_model + '/checkpoints'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory to save checkpoints and results
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        #eval_steps=10,
        #save_strategy='no',
        save_strategy='epoch',
        num_train_epochs=kwargs.get('epochs', 3),
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        label_names=['labels'],
        save_total_limit=1,
        dataloader_pin_memory=False,
        learning_rate=2e-5,
        warmup_steps=200,
    )

    def compute_metrics(pred):
        predictions, labels = pred
        if len(predictions.shape) == 1 and len(predictions) == len(labels):
            predicted_labels = (predictions > 0.5).astype(int)
        else:
            predicted_labels = np.argmax(predictions, axis=1)
        correct_predictions = (predicted_labels == labels)
        accuracy = correct_predictions.astype(float).mean().item()
        precision = precision_score(labels, predicted_labels)
        f1 = f1_score(y_true=labels, y_pred=predicted_labels)
        recall = recall_score(y_true=labels, y_pred=predicted_labels)
        return {'accuracy': accuracy, 'precision': precision, "recall": recall, "f1": f1}

    def make_state_dict_contiguous(model):
        original_state_dict = model.state_dict

        def contiguous_state_dict(*args, **kwargs):
            sd = original_state_dict(*args, **kwargs)
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and not v.is_contiguous():
                    sd[k] = v.contiguous()
            return sd

        model.state_dict = contiguous_state_dict
    make_state_dict_contiguous(model)

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if kwargs.get('save_model', False):
        trainer.save_model(output_dir=output_model)
        tokenizer.save_pretrained(output_model)

    # Step 4: Evaluate the model
    print("Evaluate")
    eval_results = trainer.evaluate(test_dataset_)
    print(eval_results)

    pred = trainer.predict(test_dataset_)

    def compute_metrics(predictions, labels, pd_test_dataset, only_basics=False):
        if len(predictions.shape) == 1 and len(predictions) == len(labels):
            predicted_labels = (predictions > 0.5).astype(int)
        else:
            predicted_labels = np.argmax(predictions, axis=1)
        labels = labels.cpu().numpy()
        correct_predictions = (predicted_labels == labels)
        accuracy = correct_predictions.astype(float).mean().item()

        # Compute precision, recall, and F1-score
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)

        if only_basics:
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n": len(pd_test_dataset),
            }

        # compute more sophisticated IR metrics
        predictions = scipy.special.softmax(predictions, axis=1)[:, 1] # prediction score (continuous) for being relevant (class label 1)

        sub_metrices = []
        for query, df in pd_test_dataset.groupby(key_lhs):
            idx = pd_test_dataset[key_lhs] == query
            idx = pd_test_dataset.index[idx].tolist()
            idx_labels = labels[idx]
            idx_predictions = predictions[idx]


            # Compute Average Precision
            average_precision = average_precision_score(idx_labels, idx_predictions)

            # Compute nDCG (Normalized Discounted Cumulative Gain)
            try:
                # see https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/metrics/_ranking.py#L1402
                ndcg = sklearn.metrics.ndcg_score([idx_labels], [idx_predictions])
            except Exception as e:
                ndcg = None

            metrics = {
                "average_precision": average_precision,
                "ndcg": ndcg
            }

            # Compute Precision at K (e.g., Precision at 10)
            sorted_indices = np.argsort(idx_predictions)[::-1]  # Sort predictions in descending order

            for k in [1, 5, 10]:
                top_k_indices = sorted_indices[:k]  # Get the top K indices
                precision_at_k = np.mean([idx_labels[i] for i in top_k_indices])
                metrics[f"precision_at_{k}"] = precision_at_k
            sub_metrices.append(metrics)
        sub_metrices = _average_dicts(sub_metrices)

        # Create a dictionary to store the metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": len(pd_test_dataset),
            **sub_metrices
        }

        return metrics

    df = test_dataset.to_pandas()
    metrics = compute_metrics(pred.predictions, test_labels, df)
    print(metrics)

    if kwargs.get('advanced_metrics', False) and 'strategy_equality' in df:
        df['id'] = list(range(len(df)))
        # create dataset which maps predictions to original data for more detailed analysis
        for strategy in ['strategy_equality', 'strategy_inequality', 'strategy_variables', 'strategy_random_formula', 'strategy_constants', 'strategy_distribute', 'strategy_swap', 'strategy_manual']:
            print("Evaluate %s" % strategy)

            # only the positive examples and the negative examples with this strategy are important
            ds = df[df[strategy]].reset_index(drop=True)
            test_ids = df['id'].isin(ds['id'])
            preds = pred.predictions[test_ids]
            labels = test_labels[test_ids]
            strategy_metrics = compute_metrics(preds, labels, ds, only_basics=True)
            metrics[strategy] = strategy_metrics

        for stat in ['strategy_count', 'is_text', 'formula_name_id', 'substituted', 'substituted_var', 'substituted_fnc']:
            unique = set(df[stat])
            sub_stats = {}
            for u in unique:
                ds = df[df[stat] == u]
                test_ids = df['id'].isin(ds['id'])
                preds = pred.predictions[test_ids]
                labels = test_labels[test_ids]
                strategy_metrics = compute_metrics(preds, labels, ds, only_basics=True)
                sub_stats[u] = strategy_metrics
            metrics[stat] = sub_stats

    # save metrics
    json.dump(metrics, open(file_path, 'w+', encoding='utf8'), indent=1)

    return metrics


def run_multiple_times(n=5, *args, **kwargs):
    metrices = []

    base_output = kwargs.get('output_model')
    for i in range(n):
        print("Start Run %d" % (i+1))

        random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        if base_output:
            kwargs['output_model'] = base_output + '/%d' % i

        kwargs['iteration'] = i

        metrics = run_ir(*args, **kwargs)
        metrices.append(metrics)
        print("Finished Run %d" % (i+1))
        print(metrics)

    averaged_metrices = _average_dicts(metrices)
    print("Averaged: %s" % averaged_metrices)

    return averaged_metrices


def run_for_input_models(models, *args, **kwargs):
    metrices = {}
    output_model = kwargs.get('output_model') or kwargs.get('output')

    base_output_model = output_model.removesuffix('/') + '/'
    for model in models:
        output = base_output_model.removesuffix('/')
        model_name = model.split('/')[-1]
        file_path = output + '/' + model_name + '.json'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if os.path.exists(file_path):
            print("File %s already exists, skip this run" % file_path)
            m = json.load(open(file_path, 'r', encoding='utf8'))
        else:
            print("Run model: %s" % model)
            if base_output_model:
                kwargs['output_model'] = base_output_model + model_name
            kwargs['input_model'] = model
            m = run_multiple_times(*args, **kwargs)
            json.dump(m, open(file_path, 'w+', encoding='utf8'), indent=1)

        metrices[model] = m
        print(metrices)

    return metrices

def run(*args, **kwargs):
    model = kwargs['model']
    print("Start formula ir run for %s" % model)
    result = run_for_input_models(models=[model], *args, **kwargs)
    print("Finished formula ir run for model %s" % model)
    print(result)
    output = kwargs.get('output')
    if output is not None:
        output = output.replace('\\', '/').removesuffix('/')
        file_path = output + '.json'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        try:
            old_result = json.load(open(file_path, 'r', encoding='utf8'))
            old_result.update(result)
            json.dump(old_result, open(file_path, 'w+', encoding='utf8'), indent=1)
        except Exception:
            json.dump(result, open(file_path, 'w+', encoding='utf8'), indent=1)
    return result

if __name__ == '__main__':
    models = '../../../models/best/'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--data', default='../../data/generated/formula-naming/NSP_temp_V1000000.csv')
    parser.add_argument('--max_size', default=5000000, type=int)
    parser.add_argument('--head', default=100000, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--n', default=2, type=int)
    parser.add_argument('--models', type=str, required=False)
    parser.add_argument('--output', type=str, default='../../models/NFIR/test')
    args = parser.parse_args()

    test = args.test
    models = args.models.split(',')
    result = run_for_input_models(models=models, data=args.data, max_size=args.max_size, head=args.head, epochs=args.epochs, n=args.n, output_model=args.output)
    print(result)
    file_path = '../../data/results/' + args.output.split('/')[-1] + '.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    json.dump(result, open(file_path, 'w+', encoding='utf8'), indent=1)