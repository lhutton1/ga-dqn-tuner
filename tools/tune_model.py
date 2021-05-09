from pathlib import Path

import tvm
from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner

from rl_tuner.ga_dqn_tuner import GADQNTuner
from rl_tuner.ga_dqn_tuner_debug import GADQNTuner as GADQNTunerDebug
from .get_model import get_model


def tune_model(mod, params, tune_settings, target, model_name):
    """
    Tune a model for a specified number of trials along with other tune settings.
    Tune settings are specified using a json configuration, as per the TVM tools readme.
    """
    early_stopping = tune_settings['early_stopping']
    number = tune_settings["number"]
    save_path = tune_settings["save_path"]
    save_name = tune_settings["save_name"]
    repeat = tune_settings["repeat"]
    debug = tune_settings.get("debug_gadqn") or False
    trials = tune_settings["trials"]
    tuner = tune_settings["tuner"]
    target = tvm.target.Target(target)

    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host="llvm",
        params=params)

    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat)

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
        elif tuner == "ga-dqn" and debug:
            tuner_obj = GADQNTunerDebug(tsk)
        elif tuner == "ga-dqn":
            tuner_obj = GADQNTuner(tsk)
        else:
            raise ValueError("invalid tuner: %s " % tuner)

        abs_path = Path(save_path + save_name).resolve()
        abs_path.mkdir(exist_ok=True, parents=True)
        abs_path_str = str(abs_path)

        tuner_obj.tune(
            n_trial=min(trials, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(trials, prefix=prefix),
                autotvm.callback.log_to_file(abs_path_str + f"/tuning_record_model={model_name}.json"),
            ],
        )

        # Save debug info for rl tuner only
        if tuner == "ga-dqn" and debug:
            tuner_obj.save_model(save_path, save_name + f"_model={model_name}_layer={i}")

        del tuner_obj


def tune_models(data):
    """
    Auto tune all models referenced in the json configuration.
    """
    target_string = data['target']
    tune_settings = data['autotuner_settings']

    for model in data['models']:
        trace, input_shapes = get_model(model['name'], model['type'])
        mod, params = relay.frontend.from_pytorch(trace, input_shapes)
        print(f"Tuning model {model['name']}, using strategy {tune_settings['tuner']}")
        tune_model(mod, params, tune_settings, target_string, model['name'])
