#!/usr/bin/env python3
import os
import json
import logging
import time

import tvm
from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner

from get_model import get_model

def tune_model(mod, params, tune_settings, target):
    early_stopping = tune_settings['early_stopping']
    number = tune_settings["number"]
    tuning_records = tune_settings["tuning_records"]
    repeat = tune_settings["repeat"]
    trials = tune_settings["trials"]
    tuner = tune_settings["tuner"]
    target = tvm.target.Target(target)

    tasks = autotvm.task.extract_from_program(
            mod["main"],
            target=target,
            target_host="llvm",
            params=params,
        )

    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat
    )

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # Create a tuner
        if tuner in ("xgb", "xgb-rank"):
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("invalid tuner: %s " % tuner)

        # If transfer learning is being used, load the existing results
        if tuning_records and os.path.exists(tuning_records):
            logging.debug("loading tuning records from %s", tuning_records)
            start_time = time.time()
            tuner_obj.load_history(autotvm.record.load_from_file(tuning_records))
            logging.debug("loaded history in %.2f sec(s)", time.time() - start_time)

        tuner_obj.tune(
            n_trial=min(trials, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(trials, prefix=prefix),
                autotvm.callback.log_to_file(tuning_records),
            ],
        )


def tune_models(data):
    target_string = data['target']
    tune_settings = data['autotuner_settings']

    for model in data['models']:
        trace, input_shapes = get_model(model['name'], model['type'])
        mod, params = relay.frontend.from_pytorch(trace, input_shapes)
        print(f"Tuning model {model['name']}, using strategy {tune_settings['tuner']}")
        tune_model(mod, params, tune_settings, target_string)
