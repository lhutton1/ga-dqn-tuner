from pathlib import Path

import tvm
from tvm import relay
from tvm.relay import testing
from tvm import autotvm

from rl_tuner.ga_dqn_tuner import DQNGATuner
from ga_tuner.ga_tuner import GATuner
from .plots import *

"""
Run experiments for reinforcement learning tuning.
"""

target = tvm.target.Target("cuda")


def run_experiments(json_config):
    """
    Run specified experiments.
    """
    config = json_config["experiments"]
    save_path = config["save_path"]
    save_name = config["save_name"]
    names = config["names"]

    if not isinstance(names, list):
        raise ValueError("names of experiments must be specified as list")

    # prevents ga/gadqn evaluation running twice when there are multiple experiments.
    has_run_trial_ga = False
    has_run_trial_gadqn = False

    # run specified experiments
    if "trial_hyperparameters" in names:
        print("Running hyperparameter trial experiment for DQN with GA.")
        trial_parameters(save_path, save_name)
    if "trial_ga" in names:
        print("Running ga tuner trial.")
        trial_ga(save_path, save_name)
        has_run_trial_ga = True
    if "trial_gadqn" in names:
        print("Running dqnga tuner trial.")
        trial_gadqn(save_path, save_name)
        has_run_trial_gadqn = True
    if "compare_gadqn_ga" in names:
        print("Comparing ga with gadqn.")
        ga_results_dir = config.get("previous_results_dir") or None
        no_trials = 10
        if not has_run_trial_gadqn:
            trial_gadqn(save_path, save_name, trials=no_trials)
        if not has_run_trial_ga and not ga_results_dir:
            trial_ga(save_path, save_name, trials=no_trials)

        compare_gadqn_with_ga(save_path, save_name,
                              expected_trials=no_trials, prev_results_dir=ga_results_dir)


def _get_relay_convolution():
    """
    Create simply relay convolution.
    """
    dtype = "float32"
    shape = (1, 3, 8, 8)
    data = relay.var("data", shape=shape, dtype=dtype)
    weight = relay.var("weight")
    out = relay.nn.conv2d(data, weight, channels=16, kernel_size=(3, 3), padding=(1, 1))
    net = relay.Function(relay.analysis.free_vars(out), out)
    return testing.create_workload(net)


def _test_convolution_with_dqnga(save_path,
                                 save_name,
                                 n_trial,
                                 early_stopping,
                                 learn_start,
                                 memory_capacity,
                                 update_frequency,
                                 discount,
                                 epsilon_max,
                                 epsilon_min,
                                 epsilon_decay,
                                 pop_size):
    """
    Test a simple convolution using RLTuner.
    """
    print(f"Running experiment with settings: n trial: {n_trial}, "
          f"early stopping: {early_stopping}, learn start: {learn_start}, "
          f"memory capacity: {memory_capacity}, update frequency: {update_frequency}, "
          f"discount: {discount}, ep decay: {epsilon_decay}")

    mod, params = _get_relay_convolution()
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=tvm.target.Target("llvm"),
        params=params)
    runner = autotvm.LocalRunner(number=1, repeat=4)
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(build_func="default"),
                                            runner=runner)
    prefix = f"[Task 1/1]"
    tuner_obj = DQNGATuner(tasks[0],
                           learn_start=learn_start,
                           memory_capacity=memory_capacity,
                           target_update_frequency=update_frequency,
                           discount=discount,
                           epsilon_decay=epsilon_decay,
                           debug=True)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)])
    tuner_obj.save_model(save_path, save_name)


def _test_convolution_with_ga(save_path,
                              save_name,
                              n_trial,
                              early_stopping,
                              pop_size):
    print(f"Running experiment with settings: n trial: {n_trial}, "
          f"early stopping: {early_stopping}, pop size: {pop_size}")

    mod, params = _get_relay_convolution()
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=tvm.target.Target("llvm"),
        params=params)
    runner = autotvm.LocalRunner(number=1, repeat=4)
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(build_func="default"),
                                            runner=runner)
    prefix = f"[Task 1/1]"
    tuner_obj = GATuner(tasks[0],
                        pop_size=pop_size,
                        debug=True)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)])
    tuner_obj.save_model(save_path, save_name)


def trial_parameters(save_path, save_name):
    """
    Run a series of experiments for DQN with GA tuner.

    Experiment Name
    ---------------
    trial_hyperparameters
    """

    # defaults
    n_trial = 2000
    early_stopping = 1e9
    learn_start = 100
    memory_capacity = 200
    update_frequency = 50
    discount = 0.99
    epsilon = (0.9, 0.05, 0.95)
    pop_size = 16

    name = save_name + "default"
    _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                 learn_start, memory_capacity, update_frequency,
                                 discount, epsilon[0], epsilon[1], epsilon[2], pop_size)

    # replay memory trails
    for ls, mc in [(100, 200), (100, 500), (100, 100), (300, 500), (100, 1000), (300, 1000), (500, 1000)]:
        name = save_name + "_learn_start=" + str(ls) + "_memory_capacity=" + str(mc)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     ls, mc, update_frequency,
                                     discount, epsilon[0], epsilon[1], epsilon[2], pop_size)

    # update frequency trials
    for uf in [1, 10, 25, 50, 100, 200]:
        name = save_name + "_update_frequency=" + str(uf)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     learn_start, memory_capacity, uf,
                                     discount, epsilon[0], epsilon[1], epsilon[2], pop_size)

    # discount trials
    for d in [0.99, 0.95, 0.85, 0.75]:
        name = save_name + "_discount=" + str(d)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     learn_start, memory_capacity, update_frequency,
                                     d, epsilon[0], epsilon[1], epsilon[2], pop_size)

    # epsilon
    for e_max, e_min, e_decay in [(1.0, 0.01, 0.99), (1.0, 0.01, 0.95), (0.9, 0.05, 0.99), (0.9, 0.05, 0.95)]:
        name = save_name + "_emax=" + str(e_max) + "_emin=" + str(e_min) + "_edecay=" + str(e_decay)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     learn_start, memory_capacity, update_frequency,
                                     discount, e_max, e_min, e_decay, pop_size)

    # pop size for ga algorithm
    for psize in [4, 8, 16, 32]:
        name = save_name + "_emax=" + str(e_max) + "_emin=" + str(e_min) + "_edecay=" + str(e_decay)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     learn_start, memory_capacity, update_frequency,
                                     discount, e_max, e_min, e_decay, psize)


def trial_ga(save_path, save_name, trials=10):
    """
    Run a number of experiments for GA tuner.

    Experiment Name
    ---------------
    trial_ga
    """
    n_trial = 2000
    early_stopping = 1e9
    pop_size = 16

    for i in range(trials):
        name = save_name + "_ga_trial=" + str(i)
        _test_convolution_with_ga(save_path, name, n_trial, early_stopping, pop_size)


def trial_gadqn(save_path, save_name, trials=10):
    """
    Run a number of experiments for GA-DQN tuner.

    Experiment Name
    ---------------
    trial_gadqn
    """

    # defaults
    n_trial = 2000
    early_stopping = 1e9
    learn_start = 100
    memory_capacity = 200
    update_frequency = 50
    discount = 0.99
    epsilon = (0.9, 0.05, 0.95)
    pop_size = 16

    for i in range(trials):
        name = save_name + "_gadqn_trial=" + str(i)
        _test_convolution_with_dqnga(save_path, name, n_trial, early_stopping,
                                     learn_start, memory_capacity, update_frequency,
                                     discount, epsilon[0], epsilon[1], epsilon[2], pop_size)


def compare_gadqn_with_ga(save_path, save_name, expected_trials, prev_results_dir=None):
    """
    Compare iterations of ga with iterations of its dqn counterpart.
    Average and log these results in a graph.

    Experiment Name
    ---------------
    compare_gadqn_ga
    """

    gadqn_tuning = []
    ga_tuning = []
    gadqn_steps = None
    ga_steps = None

    # collect best score results
    for i in range(expected_trials):
        gadqn_path = save_path + save_name + "_gadqn_trial=" + str(i)
        y_data = DynamicPlot.load(gadqn_path, "best_score").y_data
        if not gadqn_steps:
            gadqn_steps = DynamicPlot.load(gadqn_path, "best_score").x_data
        gadqn_tuning.append(y_data)
        ga_path = prev_results_dir if prev_results_dir else save_path + save_name
        ga_path = ga_path + "_ga_trial=" + str(i)
        y_data = DynamicPlot.load(ga_path, "best_score").y_data
        ga_tuning.append(y_data)
        if not ga_steps:
            ga_steps = DynamicPlot.load(ga_path, "best_score").x_data

    # create new graph displaying averages of both plots
    comparison_plot(save_path,
                    "best_score_comparison",
                    "Best score comparison",
                    "steps",
                    "best score",
                    gadqn_tuning,
                    ga_tuning,
                    gadqn_steps,
                    ga_steps)
