import comet_ml
from typing import Tuple, Optional, Union, Dict, Any, List

import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
import subprocess

class CometMLCallback(TrainerCallback):
    def __init__(self, experiment, log_interval=2):
        super().__init__()
        self.experiment = experiment
        self.log_interval = log_interval
        self.step = 0

    def on_train_batch_end(self, args, state, control, model, data_collator, trainer, **kwargs):
        # Access the loss from the trainer's state
        loss = state.log_history[-1]["loss"]

        # Log the loss using your preferred logging method (e.g., print or a logging library)
        print(f"Train Loss: {loss}")

    def on_eval_batch_end(self, args, state, control, model, data_collator, trainer, **kwargs):
        # Access the loss from the trainer's state
        loss = state.log_history[-1]["loss"]

        # Log the loss using your preferred logging method (e.g., print or a logging library)
        print(f"Eval Loss: {loss}")

    def on_step_end(self, args, state, control, **kwargs):
        # Called at the end of each training step
        self.step += 1
        if self.step % self.log_interval == 0:
            # Log custom metrics to Comet ML here
           # loss = state.log_history[-1]['loss']
            #accuracy = state.log_history[-1]['accuracy']
            #self.experiment.log_metric("loss", loss, step=self.step)
            #self.experiment.log_metric("accuracy", accuracy, step=self.step)

            try:
                # Run the nvidia-smi command and capture its output
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"])
                gpu_utilization, memory_utilization = map(float, output.decode("utf-8").strip().split(','))

                # Log GPU metrics to Comet ML
                self.experiment.log_metric("GPU Utilization", gpu_utilization, step=state.global_step)
                self.experiment.log_metric("Memory Utilization", memory_utilization, step=state.global_step)

                #print(f"GPU Utilization: {gpu_utilization}%")
                #print(f"Memory Utilization: {memory_utilization}%")

            except Exception as e:
                print(f"Error while logging GPU metrics: {str(e)}")


def precision_at_k(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
    """
    Function to compute precision@k for an input boolean dataframe

    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output

    Output:
        Floating-point number of precision value for k items
    """
    # check we have a valid entry for k
    if k <= 0:
        raise ValueError('Value of k should be greater than 1, read in as: {}'.format(k))
    # check y_test & y_pred columns are in df
    if y_test not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_test))
    if y_pred not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_pred))

    # extract the k rows
    dfK = df.head(k)
    # compute number of recommended items @k
    denominator = dfK[y_pred].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None

loss = CrossEntropyLoss()
class CustomTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)

        # Calculate the loss
        if len(outputs.shape) == 1 and len(outputs) == len(labels):
            loss_value = torch.nn.functional.binary_cross_entropy(outputs, labels.type(torch.float))
        else:
            loss_value = torch.nn.functional.cross_entropy(outputs, labels.type(torch.long))

        return (loss_value, outputs) if return_outputs else loss_value

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs
                else:
                    loss = None
                    if 'labels' in inputs:
                        inputs = {k: v for k, v in inputs.items() if k != 'labels'}

                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

def _average_dicts(data):
    if len(data) == 0:
        return {}
    sum_dict = {key: [] for key, v in data[0].items()}

    # Loop through the dictionaries and accumulate the values

    for d in data:
        for key, value in d.items():
            if value is not None:
                if not key in sum_dict:
                    sum_dict[key] = []

                sum_dict[key].append(value)

    # Calculate the average for each key
    return {key: (_average_dicts(value) if isinstance(value[0], dict) else sum(value) / len(value)) for key, value in sum_dict.items() if len(value) > 0}
