import json

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import pathlib
from transformers import AutoModel, AutoTokenizer
from itertools import groupby
from bertviz import head_view
import shutil


"""Computes the MSAS metrics. 

-model: Transformer model
- tokens: tokenized sequence to be applied to the model
- index_groups: groups indices of tokens belonging to the same mathematical symbol
- associated_tokens: associates related mathematical token indices according to the considered relation
- save_attentions: whether all aggregrated attention matrices of all attention heads should be contained in the output or not
- ignore_cls_sep: whether to ignore attentions to and from the cls and sep tokens or not for calculating the metrices (those values are always set to 0 in outputted attentions for visualization)
"""
def _compute_attention_scores(model, tokens, index_groups, associated_tokens, save_attentions=False, ignore_cls_sep=False):
    # Forward pass through the BERT model to obtain attention scores

    with torch.no_grad():
        outputs = model(**tokens, output_attentions=True)

    # Extract attention matrices for all layers and heads
    attentions = outputs['attentions']

    # Initialize a list to store attention scores for pairs of indices
    attention_scores = {}

    # Iterate through attention matrices (layers x heads)
    for layer, attention_matrix in enumerate(attentions):
        # number of attention heads in this layer (12 for bert base, 24 for bert large)
        heads = attention_matrix.shape[1]

        for head in range(heads):

            aggregated_result = {}
            if index_groups is not None and associated_tokens is not None:
                # compute accuracy
                relevant_attention = attention_matrix[0][head]
                if ignore_cls_sep:
                    cls_sep_token_indices = [i for i, t in enumerate(tokens) if t in ['[SEP]', ['[CLS]']]]
                    relevant_attention[cls_sep_token_indices, :] = 0
                    relevant_attention[:, cls_sep_token_indices] = 0

                summed_columns = [relevant_attention[:, group].sum(dim=1) for group in index_groups]
                stacked_columns = torch.stack(summed_columns, dim=1)
                aggregated_matrix = torch.stack([stacked_columns[[group]].mean(dim=0) for group in index_groups], dim=1)
                # if index_groups contains each index of relevant_attention.shape[0] (or [1] since the shapes are equal)
                # then aggregated_matrix.sum(dim=0) results in a one-tensor ([1,..., 1]) due to taking the sum and mean
                # over the corresponding dimensions. When using TestData, this property holds, iff every formula part is
                # split up separately in (a) token(s). E.g. [..., "a", "b", ...] can be tokenized to a single token "ab"
                # -> this token is duplicated here

                keys = list(associated_tokens.keys())
                values = list(associated_tokens.values())


                # [aggregated_matrix[:, key][list(range(aggregated_matrix.shape[0]))].sum() for key in associated_tokens] yields values ~ 1.0
                mean = [float(v) for key, value in associated_tokens.items() for v in aggregated_matrix[:, key][value]]


                # predictions from a word to associated words (highest attention is interpreted as prediction)
                # since multiple tokens may have the same attention value, we can't use argmax here
                # this code produces a similar output as aggregated_matrix[:, keys].argmax(dim=0) if argmax(dim=0)
                # would return a set of all args that maximize the expression

                max_value = aggregated_matrix.max(dim=0)[0]
                preds = torch.argwhere(aggregated_matrix == max_value.view(1, -1))
                evaluated = []
                for i, key in enumerate(keys):
                    maximum_indices = preds[:, 0][preds[:, 1] == key]
                    correct = any(j in values[i] for j in maximum_indices)
                    evaluated.append(correct)


                aggregated_result['accuracy'] = evaluated
                aggregated_result['mean'] = np.mean(mean)
                if save_attentions:
                    #aggregated_matrix[aggregated_matrix < 0.01] = 0
                    aggregated_result['attentions'] = aggregated_matrix.T
                    aggregated_result['accuracies'] = sum(evaluated) / len(evaluated) if len(evaluated) > 0 else 0.0
                    aggregated_result['means'] = aggregated_result['mean']
                    # create correct matrix
                    correct = torch.zeros_like(aggregated_matrix)
                    for key, values in associated_tokens.items():
                        for value in values:
                            correct[key, value] = 1
                    aggregated_result['correct'] = correct

            attention_scores[(layer, head)] = aggregated_result

    return attention_scores

"""Instance of a single formula and relation."""
class TestDataEntry:

    def __init__(self, sequence1, sequence2, mapping, equals=False, reverse=False):
        self.reverse = reverse
        self.function_mode = False
        self.single_sequence = False

        if sequence2 is None:
            self.function_mode = True
            self.single_sequence = True
        else:
            if mapping is None:
                mapping = list(range(min(len(sequence1), len(sequence2))))
            if len(mapping) < len(sequence1):
                mapping += [[]] * (len(sequence1) - len(mapping))
            elif len(mapping) == len(sequence1) + len(sequence2):
                self.function_mode = True
            elif len(mapping) > len(sequence1):
                raise ValueError("Mapping can not be larger than sequence1 (%d > %d)! The i-th position in the mapping corresponds to the indices in sequence2 that are related to the sequence1[i]." % (len(mapping), len(sequence1)))

        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.sequences = self.sequence1 if self.single_sequence else (self.sequence1 + self.sequence2)
        self.mapping = {k: v if isinstance(v, list) else [v] for k, v in enumerate(mapping)}
        self.equals = equals

    def sequence1_str(self):
        return "".join(self.sequence1)

    def sequence2_str(self):
        return "".join(self.sequence2)

    def _analyze_stats(self, sequence_array, sequence_text, offsets, token_ids):
        stats = {}
        lower_index = 0
        upper_offset_index = 0
        lower_offset_index = 0
        upper_index = offsets[upper_offset_index][1]
        sequence_index = 0

        while True:
            to_be_found_text = sequence_array[sequence_index].strip()
            text = sequence_text[lower_index:upper_index].strip()
            if to_be_found_text in text:
                while True:
                    try:
                        t = sequence_text[offsets[lower_offset_index+1][0]: upper_index].strip()
                    except Exception:
                        break
                    if to_be_found_text in t:
                        lower_offset_index += 1
                    else:
                        break

                text = sequence_text[offsets[lower_offset_index][0]: upper_index]

                token_indices = list(range(lower_offset_index, upper_offset_index+1))
                tokens = token_ids[lower_offset_index: upper_offset_index+1]

                data = {'token_indices': token_indices, 'tokens': tokens, 'to_be_found_text': to_be_found_text, 'text': text}
                stats[sequence_index] = data

                lower_offset_index = upper_offset_index
                lower_index = offsets[lower_offset_index][0]
                sequence_index += 1
                if sequence_index >= len(sequence_array):
                    break
            else:
                upper_offset_index += 1
                upper_index = offsets[upper_offset_index][1]
        return stats

    def compute_attention(self, model, tokenizer, save_attentions=False):
        if self.single_sequence:
            sequence_str = self.sequence1_str()
            tokenized_sequence = tokenizer(sequence_str, return_tensors="pt")

            tokenized_sequence1 = tokenizer(sequence_str, add_special_tokens=False, return_offsets_mapping=True)

            stats = self._analyze_stats(self.sequence1, sequence_str, tokenized_sequence1['offset_mapping'],
                                         tokenized_sequence1['input_ids'])

            # list that contains those pairs of indices wrt. tokens such that each pair corresponds to mathematically related tokens
            # e.g. corresponding to the same variable, of \frac <-> /, ...

            offset_seq = 1  # [CLS]

            # [CLS]
            index_groups = [[0]]
            for s1 in stats.values():
                index_groups.append([s + offset_seq for s in s1['token_indices']])

            # final [SEP] token
            index_groups.append([offset_seq + len(tokenized_sequence1['input_ids'])])

            def function_mode_offset(index):
                if index <= len(self.sequence1):
                    return index + offset_seq
                else:
                    return index + 2 * offset_seq

            associated_tokens = {}
            for index, mapped_indices in self.mapping.items():
                k = index + offset_seq
                v = [function_mode_offset(mi) for mi in mapped_indices]

                if len(v) == 0:
                    continue

                if k in associated_tokens:
                    associated_tokens[k] += v
                else:
                    associated_tokens[k] = v

            if self.reverse:
                associated_tokens_reversed = {}
                for k, v in associated_tokens.items():
                    if not isinstance(v, list):
                        v = [v]

                    for vv in v:
                        if vv in associated_tokens_reversed:
                            associated_tokens_reversed[vv].append(k)
                        else:
                            associated_tokens_reversed[vv] = [k]

                associated_tokens = associated_tokens_reversed


            attention_scores = _compute_attention_scores(model, tokenized_sequence, index_groups=index_groups,
                                                         associated_tokens=associated_tokens,
                                                         save_attentions=save_attentions)

            # add the tokens to the attention scores for later analysis
            if save_attentions:
                for k, v in attention_scores.items():
                    attention_scores[k]['tokens'] = ['[CLS]'] + self.sequence1 + ['[SEP]']


            return attention_scores

        else:
            sequence1_str = self.sequence1_str()
            sequence2_str = self.sequence2_str()

            if self.equals:
                tokenized_sequence = tokenizer(sequence1_str + ' = ' + sequence2_str, return_tensors="pt")
            else:
                tokenized_sequence = tokenizer(sequence1_str, sequence2_str, return_tensors="pt")

            tokenized_sequence1 = tokenizer(sequence1_str, add_special_tokens=False, return_offsets_mapping=True)
            tokenized_sequence2 = tokenizer(sequence2_str, add_special_tokens=False, return_offsets_mapping=True)

            stats1 = self._analyze_stats(self.sequence1, sequence1_str, tokenized_sequence1['offset_mapping'], tokenized_sequence1['input_ids'])
            stats2 = self._analyze_stats(self.sequence2, sequence2_str, tokenized_sequence2['offset_mapping'], tokenized_sequence2['input_ids'])

            offset_seq1 = 1  # [CLS]
            offset_seq2 = len(tokenized_sequence1['input_ids']) + 1 + offset_seq1  # [CLS] + sequence1 tokens + [SEP]

            # [CLS]
            index_groups = [[0]]
            for s1 in stats1.values():
                index_groups.append([s + offset_seq1 for s in s1['token_indices']])

            # [SEP] or = token
            index_groups.append([offset_seq2 - 1])
            for s2 in stats2.values():
                index_groups.append([s + offset_seq2 for s in s2['token_indices']])

            # final [SEP] token
            index_groups.append([offset_seq2 + len(tokenized_sequence2['input_ids'])])

            offset_seq2 = len(self.sequence1) + 1 + offset_seq1 # [CLS] + sequence1 tokens + [SEP]
            def function_mode_offset(index):
                if index <= len(self.sequence1):
                    return index + offset_seq1
                else:
                    return index + 2 * offset_seq1
            associated_tokens = {}
            for index, mapped_indices in self.mapping.items():
                k = index + offset_seq1

                mapped_indices = [m if m != '=' else (offset_seq2 - 1) for m in mapped_indices]

                if self.function_mode:
                    v = [function_mode_offset(mi) for mi in mapped_indices]
                else:
                    v = [mi + offset_seq2 for mi in mapped_indices]

                if len(v) == 0:
                    continue

                if k in associated_tokens:
                    associated_tokens[k] += v
                else:
                    associated_tokens[k] = v

            if self.reverse:
                associated_tokens_reversed = {}
                for k, v in associated_tokens.items():
                    if not isinstance(v, list):
                        v = [v]

                    for vv in v:
                        if vv in associated_tokens_reversed:
                            associated_tokens_reversed[vv].append(k)
                        else:
                            associated_tokens_reversed[vv] = [k]

                associated_tokens = associated_tokens_reversed

            attention_scores = _compute_attention_scores(model, tokenized_sequence, index_groups=index_groups, associated_tokens=associated_tokens, save_attentions=save_attentions)
            if len(set([tuple(t) for t in index_groups])) != len(index_groups):
                print("Multiple Tokens are mapped to single token: " + self.sequence1_str() + self.sequence2_str())


            # add the tokens to the attention scores for later analysis
            if save_attentions:
                for k, v in attention_scores.items():
                    if self.equals:
                        attention_scores[k]['tokens'] = ['[CLS]'] + self.sequence1 + ['='] + self.sequence2 + ['[SEP]']
                    else:
                        attention_scores[k]['tokens'] = ['[CLS]'] + self.sequence1 + ['[SEP]'] + self.sequence2 + ['[SEP]']



            return attention_scores

    def _plot_index_pairs(self, tokenized_sequence, index_pairs, tokenizer):
        sequence = [tokenizer.decode(t) for t in tokenized_sequence['input_ids'][0]]
        sequence = [list(g) for k, g in groupby(sequence[1:], lambda x: x == '[SEP]') if not k]
        sequence1 = sequence[0]
        sequence2 = sequence[1]

        unit_to_cm = 1
        # Calculate the figure size based on the desired ratio
        fig_width_cm = max(len(sequence1), len(sequence2))  # Adjust as needed
        fig_height_cm = 3 * unit_to_cm  # Adjust as needed

        # Create a Matplotlib figure with the specified size
        fig, ax = plt.subplots(figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54))  # Convert from cm to inches

        # Set the width and height of each rectangle
        rect_width = 1
        rect_height = 1

        # Set the vertical distance between the two sequences
        vertical_spacing = 4.0

        # Iterate through the tokens in the first sequence
        for i, token in enumerate(sequence1):
            x = i  # X-coordinate position of the rectangle
            y = 0  # Y-coordinate position for the first sequence
            rect = plt.Rectangle((x, y), rect_width, rect_height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.annotate(token, (x + rect_width / 2, y + rect_height / 2), color='b', fontsize=8, ha='center',
                        va='center')

        # Iterate through the tokens in the second sequence
        for i, token in enumerate(sequence2):
            x = i  # X-coordinate position of the rectangle
            y = -vertical_spacing  # Y-coordinate position for the second sequence
            rect = plt.Rectangle((x, y), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.annotate(token, (x + rect_width / 2, y + rect_height / 2), color='r', fontsize=8, ha='center',
                        va='center')

        # Draw lines connecting specific tokens (customize as needed)
        for sub_index_pairs in index_pairs:
            for i, j in sub_index_pairs:
                ax.plot([i - 0.5, j + 0.5 - len(sequence1) - 2], [0, -vertical_spacing + rect_height], color='black', linestyle='-', linewidth=1)

        # Set axis limits and labels
        ax.set_xlim(-1, max(len(sequence1), len(sequence2)))
        ax.set_ylim(-vertical_spacing - 1, 1)
        ax.axis('off')  # Turn off axis labels and ticks

        # Show the plot
        plt.show()

    def _latex(self, tokenized_sequence, index_pairs, tokenizer):
        # generate latex code
        sequence = [tokenizer.decode(t) for t in tokenized_sequence['input_ids'][0]]
        sequence = [list(g) for k, g in groupby(sequence[1:], lambda x: x == '[SEP]') if not k]
        sequence1 = sequence[0]
        sequence2 = sequence[1]

        latex = [r"\begin{tikzpicture}[node distance=-0.05cm,seq1/.style={rectangle, draw=blue!20, fill=blue!5, very thick, minimum height=7mm}, seq2/.style={rectangle, draw=red!20, fill=red!5, very thick, minimum height=7mm}]"]
        for sequence_number, sequence in zip([1, 2], [sequence1, sequence2]):
            sequence_symbol = 'upper' if sequence_number == 1 else 'lower'

            for i, t in enumerate(sequence):
                if i == 0:
                    if sequence_number == 1:
                        line = r"\node[seq%d, anchor=west] at (0,0) (%s%d) {\lstinline[style=latexstyle]|%s|};" % (sequence_number, sequence_symbol, i, t)
                    else:
                        line = r"\node[seq%d, anchor=west] at (0, -3) (%s%d) {\lstinline[style=latexstyle]|%s|};" % (sequence_number, sequence_symbol, i, t)

                else:
                    line = r"\node[seq%d, right=of %s%d] (%s%d) {\lstinline[style=latexstyle]|%s|};" % (sequence_number, sequence_symbol, i-1, sequence_symbol, i, t)
                latex.append(line)
            latex.append("")

        def swap_color(color):
            if color == 'black':
                return 'gray'
            return 'black'
        color = 'black'

        for sub_index_pairs in index_pairs:
            for i, j in sub_index_pairs:
                upper_index = i - 1
                lower_index = j - len(sequence1) - 2
                line = r"\draw[-, color=%s] (upper%d.south) -- (lower%d.north);" % (color, upper_index, lower_index)
                latex.append(line)
                #color = swap_color(color)

        latex.append(r"\end{tikzpicture}")
        print("\n".join(latex))


"Instance of a single formula with all relations."
class TestDataCluster:

    def __init__(self, data, token_restrictions=None, apply_substitution=False, equalities=False, all_functions=False):
        self.sequence1 = data['sequence1']
        self.sequence2 = data['sequence2']
        self.substitution = data['substitution']
        substituted_sequence2 = [self.substitution.get(s, s) for s in self.sequence2]
        if apply_substitution:
            self.sequence2 = substituted_sequence2

        self.mapping = data.get('mapping', list(range(min(len(self.sequence1), len(self.sequence2)))))

        offset = len(self.sequence1)
        raw_function_mapping = data['function_mapping']
        if len(raw_function_mapping) != 2:
            raise ValueError("function_mapping must consist of an 2-length array (corresponding to the two sequences)")
        elif len(raw_function_mapping[0]) != len(self.sequence1):
            raise ValueError("first function_mapping must have the length of sequence1 (%d != %d): %s" % (len(raw_function_mapping[0]), len(self.sequence1), data))
        elif len(raw_function_mapping[1]) != len(self.sequence2):
            raise ValueError("second function_mapping must have the length of sequence2 (%d != %d): %s" % (len(raw_function_mapping[0]), len(self.sequence2), data))
        def function_mapping_converter(mapping, offset):
            return [[] if isinstance(f, list) else (f if f == '=' else f + offset) for f in mapping]

        self.function_mapping = function_mapping_converter(data['function_mapping'][0], 0) + function_mapping_converter(data['function_mapping'][1], offset)
        self.function_mapping1 = function_mapping_converter(data['function_mapping'][0], 0)
        self.function_mapping2 = function_mapping_converter(data['function_mapping'][1], 0)

        self.test_data = {}

        all = TestDataEntry(self.sequence1, self.sequence2, self.mapping, equals=equalities)
        self.test_data['symbol relation'] = all


        if self.substitution:
            self.symbols_mapping = [m if s in self.substitution else [] for s, m in zip(self.sequence1, self.mapping)]
            substitution_data = TestDataEntry(self.sequence1, substituted_sequence2, self.symbols_mapping, equals=equalities) # verbose always False, since redundant to other mappings

            self.test_data['substitution relation'] = substitution_data


        if equalities:
            function_data = TestDataEntry(self.sequence1, self.sequence2, self.function_mapping, equals=equalities)
            self.test_data['function relation'] = function_data
        else:
            function_data = TestDataEntry(self.sequence1, None, self.function_mapping1, equals=equalities)
            self.test_data['function relation1'] = function_data
            function_data = TestDataEntry(self.sequence2, None, self.function_mapping2, equals=equalities)
            self.test_data['function relation2'] = function_data

        function_data = TestDataEntry(self.sequence1, self.sequence2, self.function_mapping, equals=equalities, reverse=True)
        self.test_data['function relation reversed'] = function_data


        for symbol in ['=', r'\Rightarrow']:
            if equalities:
                mapping = [len(self.sequence1)] * (len(self.sequence1) + len(self.sequence2) + 1)
                data = TestDataEntry(self.sequence1 + ['='] + self.sequence2, None, mapping, equals=equalities)
                self.test_data['='] = data
            else:
                if symbol in self.sequence1 and len([x for x in self.sequence1 if x == symbol]) == 1:
                    index = self.sequence1.index(symbol)
                    mapping = [index] * len(self.sequence1)
                    data = TestDataEntry(self.sequence1, None, mapping, equals=equalities)
                    self.test_data['='] = data

                if symbol in self.sequence2 and len([x for x in self.sequence2 if x == symbol]) == 1:
                    index = self.sequence2.index(symbol)
                    mapping = [index] * len(self.sequence2)
                    data = TestDataEntry(self.sequence2, None, mapping, equals=equalities)
                    if '=' in self.test_data:
                        self.test_data[symbol + '1'] = self.test_data.pop(symbol)
                        self.test_data[symbol + '2'] = data
                    else:
                        self.test_data[symbol] = data


        function_restrictions = {
            'function-arguments': {r'\sin', r'\cos', r'\tan'},
            'function-sin': {r'\sin'},
            'function-cos': {r'\cos'},
            'function-tan': {r'\tan'},
            'frac': {r'\frac'},
            'plus': {'+'},
            'mul-star': {'*'},
            'cdot': {r'\cdot'},
            'mul': {'*', r'\cdot', r'\times'},
            'minus': {'-'},
            'division': {'/'},
            'power': {'^'}
        }
        if not all_functions:
            # just calculate the function restrictions relevant for the master thesis
            function_restrictions = {
                'function-arguments': {r'\sin', r'\cos', r'\tan'},
                'frac': {r'\frac'},
            }

        if all_functions:
            if token_restrictions is not None:
                for token_restriction in token_restrictions:
                    restricted_mapping = [m if s in self.substitution else [] for s, m in zip(self.sequence1, self.mapping)]
                    if sum(1 if isinstance(v, int) else len(v) for v in restricted_mapping) > 0:
                        self.test_data['restricted %s' % (','.join([t.replace('\\', '') for t in token_restriction]))] = TestDataEntry(self.sequence1, self.sequence2, restricted_mapping, equals=equalities)

        if function_restrictions is not None:
            sequence = self.sequence1 + self.sequence2
            for name, function_restriction in function_restrictions.items():
                left_restricted_mapping = []
                right_restricted_mapping = []
                full_restricted_mapping = []
                for current_index, m in enumerate(self.function_mapping):
                    left_restricted_mapping_entry = []
                    right_restricted_mapping_entry = []
                    full_restricted_mapping_entry = []
                    if isinstance(m, int):
                        m = [m]
                    elif isinstance(m, str):
                        full_restricted_mapping.append([])
                        left_restricted_mapping.append([])
                        right_restricted_mapping.append([])
                        continue

                    for mapped_index in m:
                        mapped_value = sequence[mapped_index]
                        if mapped_value in function_restriction:
                            full_restricted_mapping_entry.append(mapped_index)

                            if current_index < mapped_index:
                                left_restricted_mapping_entry.append(mapped_index)
                            elif current_index > mapped_index:
                                right_restricted_mapping_entry.append(mapped_index)

                    full_restricted_mapping.append(full_restricted_mapping_entry)
                    left_restricted_mapping.append(left_restricted_mapping_entry)
                    right_restricted_mapping.append(right_restricted_mapping_entry)

                checker = lambda mapping: sum(1 if isinstance(v, int) else len(v) for v in mapping) > 0

                if checker(full_restricted_mapping):
                    def add(key, mapping, reverse=False):
                        if not equalities:
                            sub_mapping1 = mapping[:len(self.sequence1)]
                            sub_mapping2 = [[mm - len(self.sequence1) for mm in m] for m in mapping[len(self.sequence1):]]
                            check1 = checker(sub_mapping1)
                            check2 = checker(sub_mapping2)
                            if check1 and check2:
                                self.test_data[key + '1'] = TestDataEntry(self.sequence1, None, sub_mapping1, equals=equalities, reverse=reverse)
                                self.test_data[key + '2'] = TestDataEntry(self.sequence2, None, sub_mapping2, equals=equalities, reverse=reverse)
                            elif check1:
                                self.test_data[key] = TestDataEntry(self.sequence1, None, sub_mapping1, equals=equalities, reverse=reverse)
                            else:
                                self.test_data[key] = TestDataEntry(self.sequence2, None, sub_mapping2, equals=equalities, reverse=reverse)
                        else:
                            self.test_data[key] = TestDataEntry(self.sequence1, self.sequence2, mapping, equals=equalities, reverse=reverse)

                    add(name, full_restricted_mapping)

                    if all_functions:
                        add(name + ' reverse', full_restricted_mapping, reverse=True)

                        if checker(left_restricted_mapping):
                            add(name + ' left', left_restricted_mapping)
                            add(name + ' left reverse', left_restricted_mapping, reverse=True)

                        if checker(right_restricted_mapping):
                            add(name + ' right', right_restricted_mapping)
                            add(name + ' right reverse', right_restricted_mapping, reverse=True)

    def _reverse_mapping(self, mapping, sequence=None):
        mapping = [m if isinstance(m, list) else [m] for m in mapping]
        if sequence is not None:
            reverse_mapping = [[] for _ in range(len(sequence))]
        else:
            reverse_mapping = [[] for _ in range(len(mapping))]
        for i, k in enumerate(mapping):
            for kk in k:
                reverse_mapping[kk].append(i)

        return reverse_mapping

    def compute_attention(self, *args, **kwargs):
        result = {}
        for identifier, data in self.test_data.items():
            result[identifier] = data.compute_attention(*args, **kwargs)

        return result

    def sequence1_str(self):
        return "".join(self.sequence1)

    def sequence2_str(self):
        return "".join(self.sequence2)

    def name(self):
        return "[CLS] " + self.sequence1_str() + " [SEP] " + self.sequence2_str() + "[SEP]"

"Instance of all formulas and relations"
class TestData:

    def __init__(self, data, token_restrictions=None, equalities=False):
        self.cluster = []
        self.equalities = equalities
        for s in data:
            try:
                self.cluster.append(TestDataCluster(s, token_restrictions=token_restrictions, equalities=equalities))
            except NotImplementedError as e:
                raise ValueError("Could not create Testdata for <%s>\n<%s>" % (s, e))

    def compute_attention(self, *args, **kwargs):
        results = []
        for cluster in self.cluster:
            cluster_results = cluster.compute_attention(*args, **kwargs)
            results.append(cluster_results)

        return results


def _make_hashable(result, include_all_attentions=False):
    hashable = {}
    for identifier, sub_result in result.items():
        sub_dict = {}
        if isinstance(sub_result, (str, list)):
            hashable[identifier] = sub_result
            continue

        sub_dict['layer-best-mean'] = sub_result['layer-best-mean']
        sub_dict['layer-best-accuracy'] = sub_result['layer-best-accuracy']

        for metric in ['accuracy', 'mean']:
            best = {}
            best['value'] = sub_result[metric]['value']
            best['layer'] = sub_result[metric]['layer']
            best['head'] = sub_result[metric]['head']
            best['std'] = sub_result[metric]['std']
            best['std-all'] = sub_result[metric]['std-all']

            for key in ['attentions', 'correct']:
                if key in sub_result[metric]:
                    attentions_dict = {}
                    for formula, attentions in sub_result[metric][key].items():
                        attentions_dict[formula] = attentions.tolist()
            sub_dict[metric] = best

        if include_all_attentions:
            sub_hashable = {}
            for (i, j), v in sub_result.items():
                if i not in sub_hashable:
                    sub_hashable[i] = {}

                hashable_values = {}
                for k, vv in v.items():
                    if isinstance(vv, dict):
                        new_dict = {}
                        for kk, vvv in vv.items():
                            if isinstance(vvv, (list, float)):
                                new_dict[kk] = vvv
                            else:
                                new_dict[kk] = vvv.tolist()

                        hashable_values[k] = new_dict
                    else:
                        hashable_values[k] = float(vv)

                sub_hashable[i][j] = hashable_values
            sub_dict['attentions'] = sub_hashable
        hashable[identifier] = sub_dict
    return hashable

"Instance of all formulas (including sep and equ) and "
class TestDataGroup:
    def __init__(self, data=None, data_equalities=None, token_restrictions=None):
        self.test_data = []

        if data is not None:
            self.test_data.append(TestData(data, token_restrictions=token_restrictions, equalities=False))

        if data_equalities is not None:
            self.test_data.append(TestData(data_equalities, token_restrictions=token_restrictions, equalities=True))

        self.cluster = [c for data in self.test_data for c in data.cluster]

        self.print_statistics()

    def print_statistics(self):
        pass

    def compute_attention(self, model_identifier, plot=False, *args, **kwargs):
        results = []
        if plot:
            kwargs['save_attentions'] = True
        for data in self.test_data:
            result = data.compute_attention(*args, **kwargs)
            results += result

        results = self._aggregate(results)

        if '..' in model_identifier:
            model_identifier = model_identifier.replace('\\', '/').split('/')[-1]
        else:
            model_identifier = model_identifier.replace('\\', '-').replace('/', '-')


        for task, result in results.copy().items():
            best_layer_heads = []
            for metric in ['accuracy', 'mean']:
                best = max(result['head-layer'].items(), key=lambda x: x[1][metric])
                print("Task %s - %s: Layer %d, Head %d, Accuracy %.2f, Mean %.2f" % (task, metric, best[0][0] + 1, best[0][1] + 1, best[1]['accuracy'], best[1]['mean']))
                output = '../../results/html/%s/%s/%s' % (model_identifier, metric, task)
                best_layer_heads.append((best[0][0], best[0][1]))

                if plot and kwargs.get('save_attentions', False):
                    self._plot(best, output=output)


            all_accuracies_all_heads = [xx for x in result['head-layer'].values() for xx in x['accuracies'].values()]
            all_accuracies_best_head = [xx for xx in result['head-layer'][(best[0][0], best[0][1])]['means'].values()]
            all_means_all_heads = [xx for x in result['head-layer'].values() for xx in x['means'].values()]
            all_means_best_head = [xx for xx in result['head-layer'][(best[0][0], best[0][1])]['means'].values()]
            # Pearson Correlation
            std_acc_all_heads = np.std(all_accuracies_all_heads)
            std_mean_all_heads = np.std(all_means_all_heads)
            std_acc_best_head = np.std(all_accuracies_best_head)
            std_mean_best_head = np.std(all_means_best_head)
            print("Model %s, Task %s, Std Accuracy %.5f, Std Mean %.5f" % (model_identifier, task, std_acc_best_head, std_mean_best_head))
            results[task]['accuracy']['std'] = std_acc_best_head
            results[task]['mean']['std'] = std_mean_best_head
            results[task]['accuracy']['std-all'] = std_acc_all_heads
            results[task]['mean']['std-all'] = std_mean_all_heads

        return results

    def _group_strings_by_suffix_number(self, strings):
        string_dict = {}

        for s in strings:
            if s[-1].isdigit():
                key = s[:-1]

                if key not in string_dict:
                    string_dict[key] = []
                string_dict[key].append(s)

        grouped_strings = list(string_dict.values())

        return grouped_strings

    def _aggregate(self, result):
        aggregated = {}

        # aggregate results that are split up before
        flattened_results = []
        for sub_result in result:
            merge_groups = self._group_strings_by_suffix_number(list(sub_result.keys()))

            if len(merge_groups) == 0:
                flattened_results.append(sub_result)
                continue

            if not all(len(t) == 2 for t in merge_groups):
                raise ValueError

            first_keys = [m[0] for m in merge_groups]
            second_keys = [m[1] for m in merge_groups]

            sub_result1 = {k[:-1] if k in first_keys else k: v for k, v in sub_result.items() if k not in second_keys}
            sub_result2 = {k[:-1]: v for k, v in sub_result.items() if k in second_keys}
            flattened_results.append(sub_result1)
            flattened_results.append(sub_result2)

        result = flattened_results
        identifiers = set([rr for r in result for rr in r.keys()])
        stats = {}
        for identifier in identifiers:
            sub_aggregated = {}
            keys = set(rr for r in result for rr in r.get(identifier, {}).keys())
            if len(keys) == 0:
                continue

            for key in keys:
                sub_aggregated[key] = {}
                relevant_results = [r[identifier][key] for r in result if identifier in r]
                metrics = relevant_results[0].keys()
                for metric in metrics:
                    if metric in ['accuracy']:
                        all_ = [m for res in relevant_results for m in res[metric]]
                        s = float(sum(all_))  # correct predictions
                        l = len(all_)  # total predictions
                        stats[identifier] = l
                        acc = s / l
                        sub_aggregated[key][metric] = acc
                    elif metric == 'mean':
                        scores = [res[metric] for res in relevant_results]
                        sub_aggregated[key][metric] = np.mean(scores)
                    elif metric in ['attentions', 'tokens', 'accuracies', 'means', 'correct']:
                        values = {''.join(res['tokens']): res[metric] for res in relevant_results}
                        if len(values) != len(relevant_results):
                            raise ValueError
                        sub_aggregated[key][metric] = values
                    else:
                        raise ValueError("Unknown metric: %s" % metric)

            sub_aggregated = {'head-layer': sub_aggregated}

            layers = max(k[0] + 1 for k in keys)
            heads = max(k[1] + 1 for k in keys)

            for metric in ['accuracy', 'mean']:
                best_per_layer = [max([sub_aggregated['head-layer'][(layer, head)][metric] for head in range(heads)]) for layer in range(layers)]
                sub_aggregated[f'layer-best-{metric}'] = best_per_layer

                best = max(sub_aggregated['head-layer'].items(), key=lambda x: x[1]['accuracy'])
                best_layer = best[0][0]
                best_head = best[0][1]
                best_value = best[1]
                sub_aggregated[metric] = {'value': best_value[metric], 'layer': best_layer, 'head': best_head}
                s = sub_aggregated['head-layer'][(best_layer, best_head)]
                if 'attentions' in s:
                    sub_aggregated[metric]['attentions'] = s['attentions']
                if 'correct' in s:
                    sub_aggregated[metric]['attentions'] = s['correct']

            aggregated[identifier] = sub_aggregated

        print(stats)
        return aggregated

    def _plot_latex(self, best, output='../../results/structure/tex'):
        output_path = pathlib.Path(output)
        # Check if the directory already exists
        if output_path.exists():
            # Remove the contents of the existing directory (including subdirectories)
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        best_layer = best[0][0]
        best_head = best[0][1]
        accuracy = best[1]['accuracy']
        mean = best[1]['mean']
        formulas = best[1]['tokens'].keys()

        # only the best entry is provided
        for i, formula in enumerate(formulas):
            attention = best[1]['attentions'][formula]
            tokens = best[1]['tokens'][formula]
            correct = best[1]['correct'][formula]

            # set attentions of CLS and SEP to 0
            ignored_indices = [i for i, t in enumerate(tokens) if t in ['[CLS]', '[SEP]']]
            for index in ignored_indices:
                attention[index, :] = 0
                attention[:, index] = 0

            latex_content = []

            sep_indices = [i for i, t in enumerate(tokens) if t == '[SEP]']
            if len(sep_indices) == 1:
                if '=' in tokens:
                    sentence_b_start = tokens.index('=')
                else:
                    sentence_b_start = None
                continue
            else:
                start = r"""
                \begin{tikzpicture}[node distance=-0.05cm,seq1/.style={rectangle, draw=blue!20, fill=blue!5, very thick, 
                    minimum height=7mm, minimum width=5.5mm}, seq2/.style={rectangle, draw=red!20, fill=red!5, very thick, 
                    minimum height=7mm, minimum width=7.01mm},
                    >=Latex, inner sep=2pt]
                """
                latex_content = []
                latex_content.append(start)
                sequence1 = tokens[1:sep_indices[0]]
                sequence2 = tokens[sep_indices[0] + 1:-1]
                # create nodes
                for sequence_number, sequence in zip([1, 2], [sequence1, sequence2]):
                    sequence_symbol = 'upper' if sequence_number == 1 else 'lower'
                    for i, t in enumerate(sequence):
                        if i == 0:
                            if sequence_number == 1:
                                line = r"\node[seq%d, anchor=west] at (0,0) (%s%d) {\lstinline[style=latexstyle]|%s|};" % (
                                    sequence_number, sequence_symbol, i, t)
                            else:
                                line = r"\node[seq%d, anchor=west] at (0, -3) (%s%d) {\lstinline[style=latexstyle]|%s|};" % (
                                    sequence_number, sequence_symbol, i, t)
                        else:
                            line = r"\node[seq%d, right=of %s%d] (%s%d) {\lstinline[style=latexstyle]|%s|};" % (
                                sequence_number, sequence_symbol, i - 1, sequence_symbol, i, t)
                        latex_content.append(line)
                    latex_content.append("")
                # draw attention values
                color = 'blue'
                for row in range(len(sequence1)):
                    upper_index = row + 1
                    for column in range(len(sequence2)):
                        lower_index = len(sequence1) + column + 2
                        opacity = attention[lower_index, upper_index].item()
                        if opacity > 0.001:
                            line = r"\draw[-, color=%s, opacity=%s] (upper%d.south) -- (lower%d.north);" % (
                            color, opacity, row, column)
                            latex_content.append(line)
                # draw correct values
                color = 'red'
                for row in range(len(sequence1)):
                    for column in range(len(sequence2)):
                        upper_index = row + 1
                        lower_index = column + 2
                        value = correct[row, column].item()
                        if value > 0:
                            line = r"\draw[-, color=%s] (upper%d.south) -- (lower%d.north);" % (
                            color, upper_index, lower_index)
                            # latex_content.append(line)
                        # color = swap_color(color)
                latex_content.append(r"\end{tikzpicture}")


            formula_accuracy = best[1]['accuracies'][formula]
            formula_mean = best[1]['means'][formula]
            output_file = output + '/%s_%.2f.html' % (i, formula_accuracy)
            with open(output_file, 'w', encoding='utf8') as f:
                f.write(latex_content)



    """Plots the best attention with bertviz. Most importantly, we added changes such that attention values are filtered and changed the font size."""
    def _plot(self, best, results=None, output='../../results/structure/html'):
        output_path = pathlib.Path(output)
        # Check if the directory already exists
        if output_path.exists():
            # Remove the contents of the existing directory (including subdirectories)
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        best_layer = best[0][0]
        best_head = best[0][1]
        accuracy = best[1]['accuracy']
        mean = best[1]['mean']
        formulas = best[1]['tokens'].keys()
        if results is None:
            # only the best entry is provided

            for i, formula in enumerate(formulas):
                attention = best[1]['attentions'][formula]
                tokens = best[1]['tokens'][formula]
                correct = best[1]['correct'][formula].tolist()
                sep_indices = [i for i, t in enumerate(tokens) if t == '[SEP]']
                if len(sep_indices) == 1:
                    if '=' in tokens:
                        sentence_b_start = tokens.index('=')
                    else:
                        sentence_b_start = None
                else:
                    sentence_b_start = sep_indices[0]

                # set attentions of CLS and SEP to 0
                ignored_indices = [i for i, t in enumerate(tokens) if t in ['[CLS]', '[SEP]']]
                mask = torch.ones_like(attention)
                for index in ignored_indices:
                    attention[index, :] = 0
                    attention[:, index] = 0

                #attention = attention * mask

                html = head_view([attention.resize(1, 1, attention.shape[0], attention.shape[1])], tokens, sentence_b_start=sentence_b_start, layer=0, heads=[0], html_action='return') # include_layers=[0]
                #html_data = html.data.replace("style=\"font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;\"", "style=\"font-family:'Palatino', 'Palatino Linotype', 'PalatinoNova-Regular', serif;\"")
                html_data = html.data.replace("style=\"font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;\"", "style=\"font-family:'Courier New', Courier, monospace;\"")
                backup_html_data = html_data

                js = """
                            function renderCorrect(svg, attention) {

        // Add new elements
        svg.append("g")
            .attr("id", "attention2") // Container for all attention arcs
            .selectAll(".headAttention")
            .data(attention)
            .enter()
            .append("g")
            .classed("headAttention", true) // Group attention arcs by head
            .attr("head-index", (d, i) => i)
            .selectAll(".tokenAttention")
            .data(d => d)
            .enter()
            .append("g")
            .classed("tokenAttention", true) // Group attention arcs by left token
            .attr("left-token-index", (d, i) => i)
            .selectAll("line")
            .data(d => d)
            .enter()
            .append("line")
            .attr("x1", BOXWIDTH)
            .attr("y1", function () {
                const leftTokenIndex = +this.parentNode.getAttribute("left-token-index")
                return TEXT_TOP + leftTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2)
            })
            .attr("x2", BOXWIDTH + MATRIX_WIDTH)
            .attr("y2", (d, rightTokenIndex) => TEXT_TOP + rightTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2))
            .attr("stroke-width", 1)
            //.attr("opacity", 1)
            .attr("fill", "none")
            .attr("stroke", function (d) {
                return d < 0.1 ? "none": "green"; // Skip lines with attention values below the threshold
            })
            .attr("stroke-opacity", 0.8)
            .attr("stroke-dasharray", "2,4")
        ;
    }
                """

                html_data = html_data.replace('.attr("stroke-width', '.attr("fill", "none").attr("stroke-width')
                #html_data = html_data.replace('renderAttention(svg, layerAttention);', 'renderAttention(svg, layerAttention);renderCorrect(svg, layerCorrect);')
                #html_data = html_data.replace('const layerAttention = attnData.attn[config.layer_seq];', 'const layerAttention = attnData.attn[config.layer_seq];const layerCorrect = [[%s]]' % correct)
                #html_data = html_data.replace('function renderAttention(', js + '\nfunction renderAttention(')

                # remove attention values below threshold
                threshold = 0.05
                html_data = html_data.replace('svg.select("#attention").remove();', """
            svg.select("#attention").remove();
            const index_map = new Map();
            threshold = %s;
                """ % threshold).replace('.attr("left-token-index", (d, i) => i)', """
            .attr("left-token-index", (d, i) => i)
            .each(function (d, i) {
                temp_map = new Map()
                ctr = 0;
                for (let j=0; j < d.length; j++) {
                    const value = d[j];
                    if (value > %s) {
                        temp_map.set(ctr, j);
                        ctr = ctr + 1;
                    }
                }

                index_map.set(i, temp_map)
            })
                """ % threshold).replace('.attr("y2", (d, rightTokenIndex) => TEXT_TOP + rightTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2))', """
            .attr("y2", function(d, i) {
                const leftTokenIndex = +this.parentNode.getAttribute("left-token-index")
                const rightTokenIndex = index_map.get(leftTokenIndex).get(i)
                if (rightTokenIndex > -1) {
                    return TEXT_TOP + rightTokenIndex * BOXHEIGHT + (BOXHEIGHT / 2)
                }
            })
                """).replace('  updateAttention(svg)', """     
                let totalCount = 0;
                // Iterate through the outer map
                index_map.forEach((innerMap) => {
                  // Add the size of each inner map to the total count
                  totalCount += innerMap.size;
                });
                console.log("Total count of values:", totalCount);
                updateAttention(svg)
                """).replace('.append("line")', """
            .filter(d => d > %s)
            .append("line")
                """ % threshold)

                idx = html_data.index('<div')
                formula_accuracy = best[1]['accuracies'][formula]
                formula_mean = best[1]['means'][formula]
                header = "<h1>Layer %s, Head %s, Formula Accuracy %.2f Total Accuracy %.2f, Formula Mean %.3f, Total Mean %.3f</h1>" % (best_layer, best_head, formula_accuracy, accuracy, formula_mean, mean)
                html_content_all = html_data[:idx] + header + html_data[idx:]


                output_file = output + '/%s_%.2f.html' % (i, formula_accuracy)
                with open(output_file, 'w', encoding='utf8') as f:
                    f.write(html_content_all)

                html_data_img = html_data.replace('drawCheckboxes(0, svg, layerAttention);', '')
                html_data_img = html_data_img.replace('<span style="user-select:none', '<span style="user-select:none; display:none;')
                output_file = output + '/img_%s_%.2f.html' % (i, formula_accuracy)
                with open(output_file, 'w', encoding='utf8') as f:
                    f.write(html_data_img)
        else:
            # all results are provided

            for i, formula in enumerate(formulas):
                # recreate attention matrix
                attentions_dict = {k: v['attentions'][formula] for k, v in results.items()}
                max_layers = max(key[0] for key in attentions_dict.keys())
                max_heads = max(key[1] for key in attentions_dict.keys())


                attentions = []
                for layer in range(max_layers + 1):

                    heads = [v for k, v in attentions_dict.items() if k[0] == layer]  # assumes correct ordering of heads
                    tensor = torch.stack(heads)
                    tensor = tensor.resize(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
                    attentions.append(tensor)

                tokens = best[1]['tokens'][formula]
                if '[SEP]' in tokens:
                    sentence_b_start = tokens.index('[SEP]')
                else:
                    sentence_b_start = None

                html = head_view(attentions, tokens, sentence_b_start=sentence_b_start, layer=best_layer, html_action='return', heads=list(range(max_heads + 1)))  # include_layers=[0]

                idx = html.data.index('<div')
                formula_accuracy = best[1]['accuracies'][formula]
                formula_mean = best[1]['means'][formula]
                header = "<h1>Layer %s, Head %s, Formula Accuracy %s, Total Accuracy %.3f, Formula Mean %.3f</h1>" % (best_layer, best_head, formula_accuracy, accuracy, formula_mean)
                html_content = html.data[:idx] + header + html.data[idx:]

                output_file = output + '/%s_%.2f.html' % (i, formula_accuracy)
                with open(output_file, 'w+', encoding='utf8') as f:
                    f.write(html_content)


def reduce_nested_dict(d):
    reduced_dict = {}
    for key, value in d.items():
        if key == 'attention':
            continue
        if isinstance(value, dict):
            reduced_dict[key] = reduce_nested_dict(value)
        else:
            reduced_dict[key] = value
    return reduced_dict


def run(models=None, model=None, data_sep=None, data_equ=None, plot=False, output='../../results/structure.json', save_attentions=False, *args, **kwargs):
    if models is None:
        models = [model]

    if data_sep is None and data_equ is None:
        raise ValueError("Either data_sep or data_equ must not be None!")

    output_folder = '/'.join(output.replace('\\', '/').split('/')[:-1])
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    structure_data_seperator = json.load(open(data_sep, 'r+', encoding='utf8')) if data_sep else None
    structure_data_equalities = json.load(open(data_equ, 'r+', encoding='utf8')) if data_equ else None
    token_restrictions = []  #[['\\frac', '/'], ['\\binom', '\\choose']]

    test_data = TestDataGroup(data=structure_data_seperator, data_equalities=structure_data_equalities, token_restrictions=token_restrictions)

    total_result = {}

    for model_identifier in models:
        print("Process %s" % model_identifier)
        model = AutoModel.from_pretrained(model_identifier)
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        result = test_data.compute_attention(model_identifier, model=model, tokenizer=tokenizer, save_attentions=save_attentions, plot=plot)

        # result: layer -> head -> metrics
        hashable_result = _make_hashable(result, include_all_attentions=save_attentions)
        model_identifier = model_identifier.replace('\\', '/')

        output_model = output.removesuffix('.json') + '_' + model_identifier.split('/')[-1] + '.json'
        json.dump(hashable_result, open(output_model, 'w+', encoding='utf8'), indent=1)
        total_result[model_identifier] = hashable_result

    if len(models) > 0:
        # print some global statistics
        all_tasks = {x for sub_result in total_result.values() for x in sub_result.keys()}
        for metric in {'accuracy', 'mean'}:
            task_means = {task: np.mean([sub_result[task][metric]['value'] for sub_result in total_result.values() if isinstance(sub_result, dict) and isinstance(sub_result[task], dict) and isinstance(sub_result[task][metric], dict)]) for task in all_tasks}
            k = 30
            print("Top %d best tasks for %s" % (k, metric))
            for i, (task, mean) in enumerate(sorted(task_means.items(), key=lambda x: -x[1])):
                model_results = {model: sub_result[task][metric]['value'] for model, sub_result in total_result.items()}
                best_models = list(sorted(model_results.items(), key=lambda x: -x[1]))[:3]
                print("%d: Task %s, Total Mean %.3f, Best models %s" % (i+1, task, mean, best_models))
                if i >= k:
                    break


    # save attentions
    if save_attentions:
        json.dump(total_result, open(output.removesuffix('.json') + '_attentions.json', 'w+', encoding='utf8'))

    # remove attentions to reduce size significantly
    reduced_result = reduce_nested_dict(total_result)
    json.dump(reduced_result, open(output, 'w+', encoding='utf8'), indent=1)

    return total_result

def get_best_entries_in_head(head_results):
    return max(head_results.items(), key=lambda x: x[1]['accuracy'])

def get_best_entries(model_results, metric='accuracy'):
    best_entries = {}
    for task, result in model_results.items():
        layer_results = {}
        for layer, head_results in result.items():
            layer_results[layer] = get_best_entries_in_head(head_results)

        best = max(layer_results.items(), key=lambda x: x[1][1]['accuracy'])
        best_entries[task] = {'layer': int(best[0]), 'head': int(best[1][0]), 'accuracy': best[1][1][metric]}
    return best_entries

def plot(results):

    accuracies = {}
    layers = {}
    heads = {}
    models = []

    for model_identifier, model_results in results.items():
        best_entries = get_best_entries(model_results)

        models.append(model_identifier)
        for task, result in best_entries.items():
            if task not in accuracies:
                accuracies[task] = []
                heads[task] = []
                layers[task] = []

            accuracies[task].append(result['accuracy'])
            heads[task].append(result['head'])
            layers[task].append(result['layer'])

    for task in accuracies:
        if 'restricted' in task:
            continue

        fig, ax = plt.subplots()
        ax.barh(models, accuracies[task])
        for i, v in enumerate(accuracies[task]):
            label = "Accuracy=%.2f, Layer=%d, Head=%d" % (accuracies[task][i], layers[task][i], heads[task][i])
            ax.text(0.02, i, label, color='black', fontsize=14, ha='left', va='center')

        ax.set_xlabel('Accuracy')
        ax.set_title(task)
        plt.gca().set_aspect(0.1)
        plt.grid()
        plt.savefig('../../results/%s.pdf' % task)
        plt.show()



def get_models_from_path(path):
    try:
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        return subfolders
    except FileNotFoundError:
        print(f"Error: The directory '{path}' does not exist.")
        return []




if __name__ == '__main__':#
    models = get_models_from_path('../../../Masterarbeit/final/models/pretraining')
    baselines = ['AnReu/math_pretrained_bert', 'microsoft/deberta-v3-base', 'bert-base-cased',  'allenai/scibert_scivocab_cased'] # 'witiko/mathberta',


    run(models + baselines,
              data_sep='../../data/structure.json',
              data_equ='../../data/structure_equalities.json',
              plot=True,
              save_attentions=False,
              output='../../results/structure/all/structure.json'
              )
    if False:
        run(models + baselines,
                      data_sep='../../data/structure.json',
                      #data_equ='../../data/structure_equalities.json',
                      plot=False,
                      save_attentions=False,
                      output='../../results/structure/sep/structure.json'
                      )

        run(models + baselines,
                      #data_sep='../../data/structure.json',
                      data_equ='../../data/structure_equalities.json',
                      plot=False,
                      save_attentions=False,
                      output='../../results/structure/equ/structure.json'
                      )
