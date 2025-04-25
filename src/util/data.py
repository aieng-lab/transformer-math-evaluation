import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sklearn
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets



def sample_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

def create_data(data,
                max_size=None,
                data_filter=None,
                split_by_formula_name_id=False,
                start=None,
                epoch_dependent=True,
                batch_size=16,
                preprocessing=None,
                seed=0,
                n=None,
                use_challenging_falses=True,
                formula_name_id='formula_name_id',
                ):

    try:
        result = DatasetDict.load_from_disk("file://" + str(Path(data).resolve()))
    except FileNotFoundError:
        result = load_dataset(data)

    if start is not None and n is not None:
        result['train'] = result['train'].select(list(range(start, n)))

    if preprocessing:
        def mapper(example):

            for key in ['name', 'formula', 'formula1', 'formula2']:
                if key in example:
                    example[key] = preprocessing(example[key])

            return example

        for key in result:
            result[key] = result[key].map(mapper)

    if split_by_formula_name_id:
        def filter_column(df):
            columns_to_include = [col for col in train_df.columns if col != '__index_level_0__']
            return df[columns_to_include]

        # split up randomly by formula_name_id, i.e. reduce the train and test set such that there is no id overlapping between those two sets

        val_size = 0.2
        dfs = {key: result[key].to_pandas() for key in result}
        df = pd.concat(dfs.values())
        ids = list(df[formula_name_id].unique())
        random.seed(seed)
        random.shuffle(ids)

        l = len(ids)
        b1 = int(l * val_size)
        ids_val = ids[:b1]
        b2 = b1 + int(l * val_size)
        ids_test = ids[b1:b2]
        ids_train = ids[b2:]
        if len(result) == 3:
            df_train = dfs['train']
            df_test = dfs['test']
            df_val = dfs['validation']

            train_df = df_train[df_train[formula_name_id].isin(ids_train)]
            val_df = df_val[df_val[formula_name_id].isin(ids_val)]
            test_df = df_test[df_test[formula_name_id].isin(ids_test)]

            result = DatasetDict()
            result['train'] = Dataset.from_pandas(filter_column(train_df))
            result['validation'] = Dataset.from_pandas(filter_column(val_df))
            result['test'] = Dataset.from_pandas(filter_column(test_df))

        else:
            df_train = dfs['train']
            df_test = dfs['test']

            train_df = df_train[df_train[formula_name_id].isin(ids_train)]
            test_df = df_test[df_test[formula_name_id].isin(ids_test)]
            val_df = df_test[df_test[formula_name_id].isin(ids_val)]

            result = DatasetDict()
            result['train'] = Dataset.from_pandas(filter_column(train_df))
            result['test'] = Dataset.from_pandas(filter_column(test_df))
            result['validation'] = Dataset.from_pandas(filter_column(val_df))

    if data_filter:
        result = result.filter(data_filter)

    if max_size:
        new_result = DatasetDict()
        for key in result:
            ds = result[key]
            l = len(ds)
            if l > max_size:
                idx = random.choices(range(l), k=max_size)
                ds = ds.select(idx)
            new_result[key] = ds
        result = new_result

    new_dict = DatasetDict()
    # Define the number of splits (in this case, 10)

    for key in result:
        # Iterate through the splits
        dataset = result[key]

        if key == 'validation' or key == 'test':

            # make sure dataset has length of multiple of batch_size
            if len(dataset) % batch_size != 0:
                new_len = (len(dataset) // batch_size) * batch_size
                dataset = dataset.select(range(new_len))

            if 'formula1_name_id' in dataset.column_names:
                # Manually group by formula1_name_id
                groups = defaultdict(list)
                for example in dataset:
                    groups[example['formula1_name_id']].append(example)

                grouped_data = []

                for id_val, examples in groups.items():
                    group_dataset = Dataset.from_list(examples)
                    trues = group_dataset.filter(lambda x: x['label'])
                    falses = group_dataset.filter(lambda x: not x['label'])

                    if len(trues) > 0:
                        query = trues.shuffle(seed=42).select([0])
                    else:
                        raise ValueError(f"Warning: No true samples found for formula group {id_val}")

                    # Replace `formula1` in each false sample
                    modified_falses = falses.map(lambda ex: {**ex, 'formula1': query[0]['formula1']})
                    modified_trues = trues.map(lambda ex: {**ex, 'formula1': query[0]['formula1']})

                    group_combined = concatenate_datasets([modified_trues, modified_falses])
                    grouped_data.append(group_combined)

                dataset = concatenate_datasets(grouped_data)

            new_dict[key] = dataset  # use the whole dataset
            continue


        trues = dataset.filter(lambda x: x['label'])
        new_len = (len(trues) // batch_size) * batch_size
        trues = trues.select(range(new_len))

        falses = dataset.filter(lambda x: not x['label'])
        if len(trues) > 0:
            n_split = max(1, int(len(falses) / len(trues)))
        else:
            n_split = 1

        end = len(falses)
        integers = list(range(0, end))
        random.shuffle(integers)
        subset_size = max(((end // n_split) // batch_size) * batch_size, batch_size)  # floor division
        subsets = [integers[i:i + subset_size] for i in range(0, len(integers), subset_size) if
                   i + subset_size <= len(integers)]

        datasets = {}
        for i, subset in enumerate(subsets):
            false_ds = falses.select(subset)
            ds = concatenate_datasets([trues, false_ds])
            ds.shuffle(seed=42)
            datasets[i] = ds

        if len(datasets) == 1:
            datasets = datasets[0]
        elif key == 'test':
            datasets = datasets[0]

        new_dict[key] = datasets
    result = new_dict
    random_instance = random.Random(seed)

    if not use_challenging_falses:
        # replace all falses with the random trues with other split_by_formula_name_id
        key = 'train'
        key_result = result[key]

        def replace_falses(ds):
            trues = ds.filter(lambda x: x["label"])
            falses = ds.filter(lambda x: not x["label"])

            # Index trues by formula ID
            formula_to_trues = defaultdict(list)
            for ex in trues:
                formula_to_trues[ex[formula_name_id]].append(ex)

            replacement_trues = []
            for false_ex in falses:
                current_formula_id = false_ex[formula_name_id]

                # All trues from other formula groups
                candidate_trues = [
                    ex for fid, group in formula_to_trues.items()
                    if fid != current_formula_id for ex in group
                ]

                if not candidate_trues:
                    raise ValueError(f"Warning: No alternative trues found for formula group {current_formula_id}")

                replacement = random_instance.choice(candidate_trues)
                replacement['label'] = False # False as it is used now for a different formula_id
                replacement['formula_name_id'] = current_formula_id
                replacement['name'] = false_ex['name']
                replacement['strategy_random_formula'] = True
                replacement['strategy_count'] += 1
                replacement_trues.append(replacement)

            # Recombine original trues + replacement trues

            new_dataset = concatenate_datasets([trues, ds.from_list(replacement_trues)]).shuffle(seed=seed)
            return new_dataset

        # Handle split vs. dict of splits
        if isinstance(key_result, dict):
            for i in key_result:
                key_result[i] = replace_falses(key_result[i])
        else:
            result[key] = replace_falses(key_result)


    if not epoch_dependent:
        result_data = result['train'][seed % len(result['train'])]
        result['train'] = {k: result_data.shuffle(seed=i) for i, k in enumerate(result['train'].keys())}

    print("Len train = %d" % len(result['train']))
    if 'test' in result:
        print("Len test = %d" % len(result['test']))
    if 'validation' in result:
        print("Len validation = %d" % len(result['validation']))

    return result
