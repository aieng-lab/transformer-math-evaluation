import os
import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk


def generate(data, versions_per_id=250, id='formula_name_id', factor_false=10, split=(0.7, 0.15, 0.15), seed=0, strategy_threshold=200):
    print(f'Generating dataset {data} with {versions_per_id} versions per id, {factor_false} false positives, split {split}, seed {seed}')
    def enforce_strategy_thresholds_on_false(df_false, strategies, threshold, total_needed):
        used_indices = set()
        if strategies:
            selected_dfs = []

            # 1. Ensure threshold coverage per strategy
            for strat in strategies:
                s_threshold = threshold
                if strat == 'strategy_random_formula':
                    s_threshold = s_threshold * 3
                strat_df = df_false[(df_false[strat])]
                selected = strat_df.sort_values(by=id).groupby(id).head(s_threshold)
                selected_dfs.append(selected)
                used_indices.update(selected.index)

            # 2. Combine selected rows and deduplicate
            base = pd.concat(selected_dfs).drop_duplicates()
        else:
            base = df_false[df_false.index.isin(used_indices)]

        # 3. Fill up to total_needed
        for formula_id in df_false[id].unique():
            group = base[base[id] == formula_id]
            to_add = total_needed - len(group)
            print(f"Formula ID {formula_id}: {len(group)} / {total_needed} ({to_add})")
            if to_add > 0:
                remaining = df_false[(df_false[id] == formula_id) & ~df_false.index.isin(base.index)]
                filler = remaining.sort_values(by=id).groupby(id).head(to_add)
                base = pd.concat([base, filler])

        return base.sample(frac=1, random_state=seed).reset_index(drop=True)
    try:
        dsd = load_dataset(data)
    except Exception:
        dsd = load_from_disk('file://' + data)

    new_dsd = DatasetDict()
    strategy_cols = [col for col in dsd['train'].features if col.startswith('strategy_') and 'count' not in col]

    if abs(sum(split) - 1) > 0.01:
        raise ValueError(f"Split values must sum to 1. Current sum: {sum(split)}", split)

    split_counts = [int(versions_per_id * s) for s in split]
    train_n, test_n, val_n = split_counts

    df_full = dsd['train'].to_pandas()

    # Separate and shuffle
    df_true = df_full[df_full['label']].sample(frac=1, random_state=seed)
    df_false = df_full[~df_full['label']].sample(frac=1, random_state=seed)

    def create_split(df_true, df_false, n_per_id, strategy_threshold, seed_offset=0):
        strategy_threshold = int(strategy_threshold)
        true_subset = df_true.groupby(id).head(n_per_id)
        false_subset = enforce_strategy_thresholds_on_false(df_false, strategy_cols, strategy_threshold, factor_false * n_per_id)
        combined = pd.concat([true_subset, false_subset]).sample(frac=1, random_state=seed + seed_offset)
        return Dataset.from_pandas(combined.reset_index(drop=True))

    new_dsd['train'] = create_split(df_true, df_false, train_n, strategy_threshold * split[0]+1, seed_offset=0)
    new_dsd['test'] = create_split(df_true, df_false, test_n, strategy_threshold * split[1]+1, seed_offset=1)
    new_dsd['validation'] = create_split(df_true, df_false, val_n, strategy_threshold * split[2]+1, seed_offset=2)

    df = new_dsd['train'].to_pandas()
    print(df[strategy_cols].sum())
    output = 'data/' + data.split('/')[-1] + f'_{versions_per_id}'
    os.makedirs(output, exist_ok=True)
    new_dsd.save_to_disk(output)

    # Print summary
    for key in new_dsd.keys():
        df = new_dsd[key].to_pandas()
        print(f"Dataset {key} has {len(df)} rows: {df['label'].sum()} true / {len(df) - df['label'].sum()} false")
        for col in strategy_cols:
            count = df[~df['label']][col].sum()
            print(f"  {col}: {count} true (in false-label examples)")



# Example call:
#generate('ddrg/named_math_formulas') # either you generate the dataset or you use the custom fine-tuning dataset with
# additional metadata provided as ddrg/named_math_formulas_ft (with this special data, advanced metrics are computed)
generate('ddrg/math_formula_retrieval', id='formula1_name_id')