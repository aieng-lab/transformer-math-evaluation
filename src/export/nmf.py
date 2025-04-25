import os
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

from util import remove_suffix
from util.plot import init_matplotlib




model_map = {
    'AnReu/math_pretrained_bert': r'\mathPretrainedBert',
    'bert-base-cased': r'\bertbase',
    'tbs17/MathBERT': r'\mathbert',
    'tbs17/MathBERT-custom': r'\mathbertcustom',
    'models/BERT_MF_MT': r'\mamutbertmfmt',
    'models/MathBERT_MF_MT': r'\mamutmathbertmfmt',
    'models/MPBERT_MF_MT': r'\mamutmpbertmfmt',
    'ddrg/math_structure_bert': r'\mamutbert',
    'models/MathBERT_MF_MT_NMF_MFR': r'\mamutmathbert',
    'models/MPBERT_MF_MT_NMF_MFR': r'\mamutmpbert',
}

for key in list(model_map.keys()):
    if '/' in key:
        key2 = key.split('/')[-1]
        model_map[key2] = model_map[key]

MODEL_ORDER = [
    'bert-base-cased',
    'BERT_MF_MT',
    'math_structure_bert',

    'MathBERT',
    'MathBERT_MF_MT',
    'MathBERT_MF_MT_NMF_MFR',

    'math_pretrained_bert',
    'MPBERT_MF_MT',
    'MPBERT_MF_MT_NMF_MFR',
]

def plot_strategy_f1_per_identity(folder='results/nmf', metric='f1', sort=True, sorting_models=MODEL_ORDER):
    init_matplotlib(latex=True)

    data = {}

    for file in os.listdir(folder):
        if not file.endswith('.json'):
            continue
        try:
            with open(os.path.join(folder, file), 'r', encoding='utf8') as f:
                d = json.load(f)
            name = remove_suffix(file, '.json')
            data[name] = d
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not data:
        print("No data loaded.")
        return

    # Identity set
    identities = sorted(set(k for d in data.values() for k in d['formula_name_id'].keys()))
    x = list(range(len(identities)))

    # Collect metric values
    metric_values_map = {}
    for model, d in data.items():
        metric_values = [d['formula_name_id'].get(fnid, {}).get(metric, 0) for fnid in identities]
        metric_values_map[model] = metric_values

    if sort:
        matrix = np.array([values for values in metric_values_map.values()])
        means = np.mean(matrix, axis=0)
        sorted_indices = np.argsort(means)
        sorted_identities = [identities[i] for i in sorted_indices]
    else:
        sorted_indices = list(range(len(identities)))
        sorted_identities = identities

    # Plotting
    tab20c = cm.get_cmap('tab20c')

    # Define color indices for each group (3 shades from 3 color families)
    cmap = [
        tab20c(0), tab20c(1), tab20c(2),  # Blue family
        tab20c(4), tab20c(5), tab20c(7),  # Orange family
        tab20c(8), tab20c(9), tab20c(11),  # Green family
    ]
    markers = ['o', 's', '^', 'v', 'D', 'p', 'H', '*', '+', 'x', '|', '_', '1', '2', '3', '4', 'X']
    markers = ['^', 'v', '>', '+', 'x','P', 'o', 'D', 's']
    plt.figure(figsize=(14, 8))

    if sorting_models:
        # sort metric_values by occurence in sorting_models
        metric_values_map = {k: metric_values_map[k] for k in sorting_models if k in metric_values_map}


    for i, (model, metric_values) in enumerate(metric_values_map.items()):
        color = cmap[i % len(cmap)]
        marker = markers[i % len(markers)]
        sorted_values = [metric_values[j
                         ] for j in sorted_indices]
        model_label = model_map.get(model, model)
        plt.scatter(x, sorted_values, color=color, marker=marker, label=model_label, zorder=5)

    plt.grid(zorder=0)

    max_y = max(max(values) for values in metric_values_map.values())
    min_y = min(min(values) for values in metric_values_map.values())
    # round min_y to 0.1 down
    step = 0.05
    n = int(1.0/step)
    min_y = np.floor(min_y / step) / n
    max_y = np.ceil(max_y / step) / n
    print(min_y, max_y)

    plt.yticks(np.arange(min_y, max_y+step/10, step))

    def percentage_formatter(x, pos):
        return f'{100 * x:.0f}%'

    #plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Format x-axis labels
    abbreviation_mapping = {
        'Addition Theorem for Cosine': 'Addition Thm. for Cosine',
        'Addition Theorem for Tangent': 'Addition Thm. for Tangent',
        'Binomial Theorem': 'Binomial Thm.',
        'Complex Number Division': 'Complex Division',
        'Complex Number Multiplication': 'Complex Multiplication',
        'Cosine Function Definition': 'Cosine Definition',
        'Fundamental Theorem of Calculus': 'Fund. Thm. of Calculus',
        'Principle of Inclusion-Exclusion': 'Principle of Incl.-Excl.'
    }

    display_names = [abbreviation_mapping.get(name, name) for name in sorted_identities]

    step = 1 # max(1, len(display_names) // 20)  # Avoid cluttering x-axis
    display_names = [name if i % step == 0 else "" for i, name in enumerate(display_names)]
    plt.xticks(x, display_names, rotation=90)

    plt.legend(ncol=3, loc=(0.107, 1.05))
    plt.ylabel('F1-Score')
    plt.xlabel('Mathematical Identity')
    plt.tight_layout()
    plt.savefig(f'nmf-{metric}.pdf', facecolor='w', transparent=False, bbox_inches='tight')
    plt.show()


def print_strategy_accuracies(file='results/nmf.json', models=MODEL_ORDER):
    data = json.load(open(file, 'r', encoding='utf8'))
    data = {k: v for k, v in data.items() if k.split('/')[-1] in models}
    key_0 = list(data.keys())[0]
    strategies = [k for k in data[key_0].keys() if k.startswith('strategy_') and 'count' not in k]

    mean_values = {strategy: np.mean([float(data[k][strategy]['accuracy']) for k in data.keys()]) for strategy in strategies}
    print(mean_values)
    sorted_strategies = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)
    print("Strategies sorted by mean accuracy:")
    for strategy, mean in sorted_strategies:
        print(f"{strategy}: {mean:.4f}")

    # report number of samples per strategy
    n_samples = {strategy: np.mean([data[k][strategy]['n'] for k in data.keys()]) for strategy in strategies}
    print("Number of samples per strategy:")
    for strategy, n in n_samples.items():
        print(f"{strategy}: {n:.4f}")

    # compute correlation between accuracy and number of samples
    x = np.array([n_samples[strategy] for strategy in strategies])
    y = np.array([mean_values[strategy] for strategy in strategies])
    correlation = np.corrcoef(x, y)[0, 1]
    print(f"Correlation between accuracy and number of samples: {correlation:.4f}")

def print_strategy_count_accuracies(file='results/nmf.json', max_count=5, models=MODEL_ORDER):
    data = json.load(open(file, 'r', encoding='utf8'))
    data = {k: v for k, v in data.items() if k.split('/')[-1] in models}
    key = 'strategy_count'

    counts = set.union(*[set(v[key].keys()) for v in data.values()])

    mean_values = {count: [float(v[key][count]['accuracy']) for v in data.values()] for count in counts}
    n_counts = {count: np.mean([v[key][count]['n'] for v in data.values() if count in v[key]]) for count in counts}

    for count in counts:
        if int(count) > max_count:
            print(f'Merging {count} into {max_count} because it is larger than max_count {max_count}')
            mean_values[str(max_count)] += mean_values[count]
            n_counts[str(max_count)] += n_counts[count]
    counts = {count for count in counts if int(count) <= max_count}

    # compute means
    mean_values = {count: np.mean(values) for count, values in mean_values.items()}

    # print table with mean_values and n_counts
    for count in sorted(counts):
        count_label = count
        if int(count) == max_count:
            count_label = f'>{count}'
        print(f'{count_label}: {mean_values[count]:.4f} ({n_counts[count]})')

def print_substituted_scores(file='results/nmf.json',
                             #metrics=('accuracy', 'precision', 'recall', 'f1'),
                             metrics=('accuracy',),
                             models=MODEL_ORDER):
    data = json.load(open(file, 'r', encoding='utf8'))
    data = {k: v for k, v in data.items() if k.split('/')[-1] in models}
    key_0 = list(data.keys())[0]
    substituted_variants = [k for k in data[key_0].keys() if k.startswith('substituted')]
    mean_values_true = {sub: {metric: np.mean([float(v[sub]['true'][metric]) for v in data.values()]) for metric in metrics} for sub in substituted_variants}
    mean_values_false = {sub: {metric: np.mean([float(v[sub]['false'][metric]) for v in data.values()]) for metric in metrics} for sub in substituted_variants}
    n_true = {sub: np.mean([v[sub]['true']['n'] for v in data.values()]) for sub in substituted_variants}
    n_false = {sub: np.mean([v[sub]['false']['n'] for v in data.values()]) for sub in substituted_variants}

    for sub in substituted_variants:
        print(f"Substituted {sub}:")
        for metric in metrics:
            true_value = mean_values_true[sub][metric]
            false_value = mean_values_false[sub][metric]
            print(f"  {metric}: {true_value:.4f} (true), {false_value:.4f} (false), n true: {n_true[sub]}, n false: {n_false[sub]}")

    print("\\begin{tabular}{l" + "cc" * len(metrics) + "}")
    print("\\toprule")
    header = "Variant & " + " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{{metric.capitalize()}}}" for metric in metrics]) + " \\\\"
    print(header)
    cmidrule = " & " + " & ".join([f"True & False" for _ in metrics]) + " \\\\"
    print("\\cmidrule(lr){2-3} " + " ".join(
        [f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(1, len(metrics))]))
    print(cmidrule)
    print("\\midrule")

    for sub in substituted_variants:
        row = [sub]
        for metric in metrics:
            true_value = mean_values_true[sub][metric]
            false_value = mean_values_false[sub][metric]
            row += [f"{true_value:.4f}", f"{false_value:.4f}"]
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


def print_text_analysis(file='results/nmf.json', models=MODEL_ORDER):
    data = json.load(open(file, 'r', encoding='utf8'))
    data = {k: v for k, v in data.items() if k.split('/')[-1] in models}

    # Print header
    header = ['True Accuracy', 'False Accuracy', 'True N', 'False N']
    print(' & '.join(header))

    # Print metric values averaged over all models
    is_texts = [d['is_text'] for d in data.values()]
    averaged_metrics = {k: {kk: np.mean([v[k][kk] for v in is_texts]) for kk in is_texts[0]['true'].keys()} for k in is_texts[0].keys()}
    row = []
    true_acc = averaged_metrics['true']['accuracy']
    false_acc = averaged_metrics['false']['accuracy']

    row.append(f"{true_acc:.4f}")
    row.append(f"{false_acc:.4f}")
    row.append(f"{averaged_metrics['true']['n']:.4f}")
    row.append(f"{averaged_metrics['false']['n']:.4f}")
    print(' & '.join(row) + r' \\')

    # print metric per model
    for model_name, model_metrics in data.items():
        row = []
        for k in ['true', 'false']:
            row.append(f"{model_metrics['is_text'][k]['accuracy']:.4f}")
            row.append(f"{model_metrics['is_text'][k]['n']:.4f}")
        # compute accuracy difference
        diff = model_metrics['is_text']['true']['accuracy'] - model_metrics['is_text']['false']['accuracy']
        row.append(f"{diff:.4f}")
        print(model_name + ' & ' + ' & '.join(row) + r' \\')

def print_nmf_analysis(file='results/nmf.json', metrics=('precision', 'recall', 'f1', 'precision_at_1', 'precision_at_10', 'average_precision', 'ndcg')):
    data = json.load(open(file, 'r', encoding='utf8'))

    # Print header
    header = ['Model'] + list(metrics)
    print(' & '.join(header))

    # Print metric values for each model
    for model_name, model_metrics in data.items():
        row = [model_map.get(model_name, model_name)]
        for metric in metrics:
            value = model_metrics.get(metric, '-')
            # Format floats to 4 decimal places
            if isinstance(value, float):
                value = f"{value*100:.1f}"
            row.append(str(value))
        print(' & '.join(row) + r' \\')

def print_mfr_analysis():
    print_nmf_analysis(file='results/mfr.json')

def print_nmf_split_analysis():
    print_nmf_analysis(file='results/nmf-split.json')

if __name__ == '__main__':
    print_strategy_accuracies()
    #print_strategy_count_accuracies()
    print_substituted_scores()
    #print_text_analysis()

    #plot_strategy_f1_per_identity()
    #print_nmf_analysis()
    #print_mfr_analysis()

    #print_nmf_split_analysis()