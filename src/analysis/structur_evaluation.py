import json

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from scipy.stats import pearsonr, spearmanr


def analyze():
    stats = {}
    data_keys = ['all'] # , 'sep', 'equ'
    for key in data_keys:
        folder = '../../results/structure/%s/' % key
        data = json.load(open(folder + 'structure.json', 'r', encoding='utf8'))
        stats[key] = data

    models = stats['all'].keys()
    table = []
    for model in models:

        values = []
        for task in ['symbol relation', 'substitution relation', '=', 'frac', 'function-arguments']:
            for metric in ['accuracy', 'mean']:
                for key in ['all', 'sep', 'equ']:
                    try:
                        data = stats[key][model][task][metric]
                        values.append(data['value'])
                    except Exception:
                        values.append(None)
        #print("%s: %s" % (model, values))
        table.append([model, *values])
    headers = ['Sym Acc All', 'Sym Acc SEP', 'Sym Acc Equ', 'Sym Mean All', 'Sym Mean Sep', 'Sym Mean Equ',
               'Sub Acc All', 'Sub Acc SEP', 'Sub Acc Equ', 'Sub Mean All', 'Sub Mean Sep', 'Sub Mean Equ',
               '= Acc All', '= Acc SEP', '= Acc Equ', '= Mean All', '= Mean Sep', '= Mean Equ',
               'frac Acc All', 'frac Acc SEP', 'frac Acc Equ', 'frac Mean All', 'frac Mean Sep', 'frac Mean Equ',
               'fnc-a Acc All', 'fnc-a Acc SEP', 'fnc-a Acc Equ', 'fnc-a Mean All', 'fnc-a Mean Sep', 'fnc-a Mean Equ']
    print(tabulate(table, headers=headers))


    cmap = plt.get_cmap('tab20')
    # plot
    for i, model in enumerate(models):

        for task in ['symbol relation']:

            for key in data_keys:
                accuracy = stats[key][model][task]['accuracy']['value']
                mean = stats[key][model][task]['mean']['value']

                plt.scatter(accuracy, mean, c=cmap(i))
    plt.show()

    try:
        for i, model in enumerate(models):

            seps = []
            equs = []
            for task in ['symbol relation', 'substitution relation']:

                sep = stats['sep'][model][task]['accuracy']['value']
                equ = stats['equ'][model][task]['accuracy']['value']
                seps.append(sep)
                equs.append(equ)

            plt.scatter(seps, equs, c=cmap(i), label=model)
        plt.legend(fontsize=7, loc=(1, 0))
        plt.show()
    except Exception:
        pass

    # analyze layer-head prediction agreement of accuracy and mean
    tasks = ['substitution relation', 'symbol relation', '=', 'function-arguments', 'frac']
    accuracies = []
    means = []
    key = 'all'
    for task in tasks:
        for model in models:

            accuracy = stats[key][model][task]['accuracy']
            mean = stats[key][model][task]['mean']
            accuracies.append((accuracy['layer'], accuracy['head']))
            means.append((mean['layer'], mean['head']))

    aggree = [all(x) for x in np.equal(accuracies, means)]
    proportion = np.array(aggree).mean()
    print("Accuracy and Mean aggree on %d/%d cases (%.1f%%)" % (sum(aggree), len(aggree), 100 * proportion))


    # analyze whether the accuracy and mean lead to different rankings
    for task in tasks:
        accuracies = []
        means = []
        for model in models:
            accuracy = stats[key][model][task]['accuracy']
            mean = stats[key][model][task]['mean']
            accuracies.append(accuracy['value'])
            means.append(mean['value'])

        print(task)
        pearson_coeff = pearsonr(accuracies, means)
        print(pearson_coeff)
        spearman_coeff = spearmanr(accuracies, means)
        print(spearman_coeff)
        corr = np.corrcoef(means, accuracies)[0, 1]
        print(corr)

    # evaluate stds
    for task in tasks:
        try:
            stds_a = [stats[key][model][task]['accuracy']['std'] for model in models if 'std' in stats[key][model][task]['accuracy']]
            stds_m = [stats[key][model][task]['mean']['std'] for model in models]
            print("%s: Mean std accuracy %.3f, Mean std mean %.3f" % (task, np.mean(stds_a), np.mean(stds_m)))
        except Exception:
            pass

    # evaluate stds
    print("-"*50)
    for task in tasks:
        try:
            stds_a = [stats[key][model][task]['accuracy']['std-all'] for model in models if 'std' in stats[key][model][task]['accuracy']]
            stds_m = [stats[key][model][task]['mean']['std-all'] for model in models]
            print("%s: Mean std accuracy %.3f, Mean std mean %.3f" % (task, np.mean(stds_a), np.mean(stds_m)))
        except Exception:
            pass


analyze()
