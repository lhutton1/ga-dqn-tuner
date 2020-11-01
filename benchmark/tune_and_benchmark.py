import json
import os
import logging
import time

import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm import rpc
from tvm.contrib import util

from tune_model import tune_model
from benchmark_model import compile_model, run_model
from get_model import get_model

if __name__ == "__main__":
    with open('config.json') as json_file:
        data = json.load(json_file)
        target_string = data['target']
        tune_settings = data['autotuner_settings']

        for model in data['models']:
            trace, input_shapes = get_model(model['name'], model['type'])
            mod, params = relay.frontend.from_pytorch(trace, input_shapes)
            print(f"Tuning model {model['name']}")
            tune_model(mod, params, tune_settings, target_string)

        run_settings = data['run_settings']

        for model in data['models']:
            for apply_tuning in [True, False]:
                trace, input_shapes = get_model(model['name'], model['type'])
                mod, params = relay.frontend.from_pytorch(trace, input_shapes)

                print(f"Compiling model {model['name']}")
                if apply_tuning:
                    graph, lib, params = compile_model(mod, params, target_string, tune_settings['tuning_records'])
                else:
                    graph, lib, params = compile_model(mod, params, target_string)
                print(f"Running model {model['name']}")
                if apply_tuning:
                    run_model(graph, lib, params, run_settings, model['name'], tune_settings['tuning_records'])
                else:
                    run_model(graph, lib, params, run_settings, model['name'])
