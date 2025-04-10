import random

import pandas as pd
import sklearn
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets




def sample_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)

def create_data(data, max_size=None, data_filter=None, split_by_formula_name_id=False, head=None, classification=False, start=None, epoch_dependant=True, batch_size=16, preprocessing=None, seed=0, n=None):
    if 'split' in data:
        ds = Dataset.load_from_disk(data + '/%d' % n)
        result = DatasetDict()
        result['train'] = ds
        result['test'] = Dataset.load_from_disk(data + '/test')
        if n % 10 != 0:
            result['test'] = result['test'].select(list(range(32)))
    elif data.endswith('.txt'):
        raw_dataset = load_dataset('text', data_files=data)['train']
        raw_dataset = raw_dataset.shuffle()
        if max_size:
            raw_dataset = raw_dataset.filter(lambda example, idx: idx < max_size, with_indices=True)

        raw_dataset = sklearn.model_selection.train_test_split(raw_dataset, train_size=0.8, shuffle=True)
        result = DatasetDict()
        result['train'] = Dataset.from_dict(raw_dataset[0])
        result['test'] = Dataset.from_dict(raw_dataset[1])
    elif data.endswith('.csv'):
        from sklearn.model_selection import train_test_split
        val_size = 0.15
        test_size = 0.15
        # 'test_size' and 'val_size' are the fractions of the data to allocate for test and validation sets
        df = pd.read_csv(data)

        if max_size and max_size < len(df):
            df = sample_dataframe(df)
            df = df.head(max_size)

        if classification:
            name_ids = df['formula_name_id'].unique()
            label2id = {name: i + 1 for i, name in enumerate(sorted(name_ids))}
            df['label'] = df.apply(lambda row: 0 if not row['label'] else label2id[row['formula_name_id']], axis=1)

        if head:
            df = df.sample(frac=1).reset_index(drop=True)
            if 'formula_name_id' in df:
                df = df.groupby('formula_name_id').head(head)
            elif 'formula1_name_id' in df:
                df = df.groupby('formula1_name_id').head(head)
            else:
                raise ValueError()


        if split_by_formula_name_id:
            ids = list(df['formula_name_id'].unique())
            random.shuffle(ids)
            l = len(ids)
            b1 = int(l * val_size)
            ids_val = ids[:b1]
            b2 = b1 + int(l * val_size)
            ids_test = ids[b1:b2]
            ids_train = ids[b2:]

            train_df = df[df['formula_name_id'].isin(ids_train)]
            val_df = df[df['formula_name_id'].isin(ids_val)]
            test_df = df[df['formula_name_id'].isin(ids_test)]
        else:
            train_df, temp_df = train_test_split(df, test_size=test_size + val_size)
            val_df, test_df = train_test_split(temp_df, test_size=val_size / (test_size + val_size))

        result = DatasetDict()
        result['train'] = Dataset.from_pandas(train_df)
        result['validation'] = Dataset.from_pandas(val_df)
        result['test'] = Dataset.from_pandas(test_df)

        try:
            test_df['strategy_count'] = test_df[['strategy_equality', 'strategy_manual', 'strategy_inequality', 'strategy_variables', 'strategy_random_formula', 'strategy_constants', 'strategy_distribute', 'strategy_swap']].sum(axis=1)
        except KeyError:
            pass


    else:
        result = DatasetDict.load_from_disk(data)

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
            ids = list(df['formula_name_id'].unique())
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
                df_val = dfs['val']

                train_df = df_train[df_train['formula_name_id'].isin(ids_train)]
                val_df = df_val[df_val['formula_name_id'].isin(ids_val)]
                test_df = df_test[df_test['formula_name_id'].isin(ids_test)]

                result = DatasetDict()
                result['train'] = Dataset.from_pandas(filter_column(train_df))
                result['validation'] = Dataset.from_pandas(filter_column(val_df))
                result['test'] = Dataset.from_pandas(filter_column(test_df))

            else:
                df_train = dfs['train']
                df_test = dfs['test']

                train_df = df_train[df_train['formula_name_id'].isin(ids_train)]
                test_df = df_test[df_test['formula_name_id'].isin(ids_test)]
                val_df = df_test[df_test['formula_name_id'].isin(ids_val)]

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

    if epoch_dependant:
        new_dict = DatasetDict()
        # Define the number of splits (in this case, 10)

        for key in result:
            # Iterate through the splits
            dataset = result[key]

            if key == 'validation':
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
                datasets = datasets[0] # choose random validation set



            new_dict[key] = datasets
        result = new_dict

    print("Len train = %d" % len(result['train']))
    if 'test' in result:
        print("Len test = %d" % len(result['test']))
    if 'validation' in result:
        print("Len validation = %d" % len(result['validation']))

    return result
