import argparse
import time
from datetime import timedelta

import humanize

from finetuning.formula_ir import run as formula_ir_run
try:
    from finetuning.arqmath.arqmath import run as arqmath_run, run_eval as arqmath_eval
except ImportError:
    pass
from analysis.structure import run as structure_run

import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-config', type=str, required=False, default='../config/default.json')
    #parser.add_argument('-comet', type=str, required=False, help='Comet ML API Key')
    parser.add_argument('-data_dir', type=str, default='', help='extern data directory')
    parser.add_argument('-output_dir', type=str, default='results/')
    parser.add_argument('-part', type=int)
    args = parser.parse_args()
    return args

run_mapping = {
    'formula-ir': formula_ir_run,
    'math-structure-score': structure_run,
    #'arqmath': arqmath_run,
    #'arqmath-eval': arqmath_eval,
}

from transformers import logging as hf_logging

# Set the logging level for the Hugging Face `transformers` library
hf_logging.set_verbosity_info()


if __name__ == '__main__':
    args = parse_args()

    config_file = args.config
    if config_file.endswith('.json'):
        # the config is given as file
        file_name = config_file
        config = json.load(open(file_name, 'r', encoding='utf8'))
    else:
        raise ValueError("Config must be a json file, got <%s>" % config_file)

    if 'runs' in config:
        base_config = {k: v for k, v in config.items() if k != 'runs'}

    else:
        # a single run is provided
        run_name = config_file.replace('\\', '/').split('/')[-1].removesuffix('.json')
        base_config = {}
        config = {'runs': {run_name: config}}

    model = args.model
    ext_data_dir = args.data_dir.removesuffix('/') + '/' # make sure the path ends with /
    int_data_dir = '../data/'
    output_dir = args.output_dir.removesuffix('/') + '/'

    if args.part is not None:
        base_config['part'] = args.part

    print("Model %s config %s" % (model, config_file))

    for name, run_config in config['runs'].items():
        id = run_config.pop('id')
        if id not in run_mapping:
            raise ValueError("Got unknown run id <%s>, available ids are %s" % (id, list(run_mapping.keys())))
        run_method = run_mapping[id]

        # fill in default values of config if used for this run method
        for k, v in base_config.items():
            if k not in run_config:
                run_config[k] = v

        if 'output' not in run_config:
            run_config['output'] = output_dir + name + '/'

        data_loc = run_config.get('data_loc', '')
        if 'ext' in data_loc:
            # data is in a folder outside the container
            data_dir = ext_data_dir
        elif 'int' in data_loc:
            data_dir = int_data_dir
        else:
            # use default values depending on the id
            if id in ['math-structure-score']:
                data_dir = int_data_dir
            else:
                data_dir = ext_data_dir

        for data in ['data', 'data_sep', 'data_equ']:
            if data in run_config:
                run_config[data] = data_dir + run_config[data]


        for key, value in run_config.copy().items():
            if not isinstance(value, str):
                continue

            if value.lower() == 'true':
                run_config[key] = True
            elif value.lower() == 'false':
                run_config[key] = False
            elif value.startswith('/'):
                run_config[key] = data_dir + value

        print("Run: %s - %s" % (name, id))
        print("Config:")
        for k, v in run_config.items():
            print(" %s: %s" % (k, v))
        start = time.time()

        run_method(model=model, **run_config)

        end = time.time()
        diff = end - start
        print("Finished run <%s> in %s (%ss)" % (id, humanize.naturaldelta(timedelta(seconds=diff)), diff))




