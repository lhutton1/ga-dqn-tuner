import numpy as np

import tvm
from tvm import relay
from tvm.relay import testing
from tvm import autotvm

from rl_tuner.ga_dqn_tuner_debug import GADQNTuner, RewardFunction
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

    # get experiments to run
    names = config["names"]
    if not isinstance(names, list):
        raise ValueError("names of experiments must be specified as list")

    # get workloads to run experiments on. Default is C1
    c1_workload = {
        "name": "convolution1",
        "type": "op"
    }
    workloads = json_config.get("models") or [c1_workload]

    for workload in workloads:
        # prevents ga/gadqn evaluation running twice when there are multiple experiments.
        has_run_trial_ga = False
        has_run_trial_gadqn = False

        # run specified experiments
        if "trial_hyperparameters" in names:
            print(f"Running hyperparameter trial experiment for DQN with GA on {workload['name']}.")
            trial_parameters(save_path, save_name, workload)
        if "trial_ga" in names:
            print(f"Running ga tuner trial on {workload['name']}.")
            trial_ga(save_path, save_name, workload)
            has_run_trial_ga = True
        if "trial_gadqn" in names:
            print(f"Running dqnga tuner trial {workload['name']}.")
            trial_gadqn(save_path, save_name, workload)
            has_run_trial_gadqn = True
        if "compare_gadqn_ga" in names:
            print(f"Comparing ga with gadqn {workload['name']}.")
            ga_results_dir = config.get("previous_results_dir") or None
            no_trials = 10
            if not has_run_trial_gadqn:
                trial_gadqn(save_path, save_name, workload, trials=no_trials)
            if not has_run_trial_ga and not ga_results_dir:
                trial_ga(save_path, save_name, workload, trials=no_trials)

            compare_gadqn_with_ga(save_path, save_name, workload,
                                expected_trials=no_trials, prev_results_dir=ga_results_dir)
        if "compare_reward" in names:
            print(f"Comparing reward functions {workload['name']}.")
            no_trials = 5
            for reward_function in RewardFunction:
                trial_gadqn(save_path, save_name + f"_reward={reward_function}", workload, 
                            trials=no_trials, reward_function=reward_function)
            trial_ga(save_path, save_name, workload, trials=no_trials)

            compare_reward_with_ga(save_path, save_name, workload, expected_trials=no_trials)


def _get_relay_workload(workload):
    """
    Get the workload determined by the JSON configuration 'model' parameter.
    If no such workload is found, or the paramter value specified is not a 
    single workload, then an exception is raised.
    """
    assert workload["type"] == "op", "Only single workloads are supported in experiments."
    workload_name = workload["name"]

    if workload_name == "convolution1":
        dtype = "float32"
        shape = (1, 144, 28, 28)
        data = relay.var("data", shape=shape, dtype=dtype)
        weight = relay.var("weight")
        out = relay.nn.conv2d(data, weight, channels=32, kernel_size=(1, 1))
        net = relay.Function(relay.analysis.free_vars(out), out)
        return testing.create_workload(net)
    elif workload_name == "convolution2":
        dtype = "float32"
        shape = (20, 16, 50, 100)
        data = relay.var("data", shape=shape, dtype=dtype)
        weight = relay.var("weight")
        out = relay.nn.conv2d(data, weight, channels=32, kernel_size=(3, 3))
        net = relay.Function(relay.analysis.free_vars(out), out)
        return testing.create_workload(net)
    elif workload_name == "matmul1":
        dtype = "float32"
        data = relay.var("data", shape=(100, 30, 40), dtype=dtype)
        data2 = relay.var("data", shape=(1, 50, 40), dtype=dtype)
        out = relay.nn.batch_matmul(data, data2)
        net = relay.Function(relay.analysis.free_vars(out), out)
        return testing.create_workload(net)
    elif workload_name == "matmul2":
        dtype = "float32"
        data = relay.var("data", shape=(30, 30, 30), dtype=dtype)
        data2 = relay.var("data", shape=(30, 30, 30), dtype=dtype)
        out = relay.nn.batch_matmul(data, data2)
        net = relay.Function(relay.analysis.free_vars(out), out)
        return testing.create_workload(net)
    else:
        raise ValueError(f"Workload name {workload_name} not recognised.")


def _test_op_with_dqnga(save_path,
                        save_name,
                        workload_name,
                        n_trial,
                        early_stopping,
                        learn_start,
                        update_frequency,
                        train_frequency,
                        discount,
                        epsilon_decay,
                        agent_batch_size,
                        hidden_sizes,
                        learning_rate,
                        reward_function=RewardFunction.R3):
    """
    Test a specified single workload using RLTuner.
    """
    print(f"Running experiment with settings: n trial: {n_trial}, "
          f"early stopping: {early_stopping}, learn start: {learn_start}, "
          f"update frequency: {update_frequency}, discount: {discount}, "
          f"ep decay: {epsilon_decay}, hidden sizes: {hidden_sizes},"
          f"agent batch size: {agent_batch_size}, learning rate: {learning_rate}")

    mod, params = _get_relay_workload(workload_name)
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=tvm.target.Target("llvm"),
        params=params)
    runner = autotvm.LocalRunner(number=1, repeat=4)
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(build_func="default"),
                                            runner=runner)
    prefix = f"[Task 1/1]"
    tuner_obj = GADQNTuner(tasks[0],
                           learn_start=learn_start,
                           target_update_frequency=update_frequency,
                           train_frequency=train_frequency,
                           discount=discount,
                           epsilon_decay=epsilon_decay,
                           agent_batch_size=agent_batch_size,
                           hidden_sizes=hidden_sizes,
                           learning_rate=learning_rate,
                           reward_function=reward_function)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)])
    tuner_obj.save_model(save_path, save_name)


def _test_op_with_ga(save_path,
                     save_name,
                     workload_name,
                     n_trial,
                     early_stopping):
    """
    Test a specified single workload with GA tuner.
    """
    print(f"Running experiment with settings: n trial: {n_trial}, "
          f"early stopping: {early_stopping}")

    mod, params = _get_relay_workload(workload_name)
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
                        debug=True)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)])
    tuner_obj.save_model(save_path, save_name)


def _generate_trials(space, r_factor=3):
    """Generates a series of trials.
    This algorithm generates a series of non-deterministic trials given a
    space of options to test. A trial is generated by pulling a value from
    each option in the space. On some occasions the values are shuffled to
    ensure a different trial on each r_factor iteration. The algorithm ensures
    that each value from an option is used at least once. The total number of
    trials is determined by the r_factor * the option with the largest number
    of values.

    NOTE: This function is from another project I worked on. It was written by me,
    however, it is not unique to this project. It can originally be found here:
    https://github.com/apache/tvm/commit/c958bc17038728f396ef94609c4a98462d545390#diff-7202aa3d01e2a8a4e6549d80b594c3b32c7af84e5965c3ebd957966946983c91R280-R319
    """
    # seed for deterministic 'randomness', produces same trials each time.
    np.random.seed(0)
    max_len = 1
    for option in space:
        max_len = max(max_len, len(option))
    num_trials = r_factor * max_len
    trials = []
    for i in range(num_trials):
        trial = []
        for option in space:
            if i % len(option) == 0:
                np.random.shuffle(option)
            trial.append(option[i % len(option)])
        trials.append(trial)
    return trials


def trial_parameters(save_path, save_name, workload):
    """
    Run a series of experiments for DQN with GA tuner.

    A random search is used (with some modification), as a grid search
    would be too expensive due to the number of hyper-parameters. A typical
    random search has been modified in that each hyper-parameter is
    guaranteed to be used.

    Experiment Name
    ---------------
    trial_hyperparameters
    """
    n_trial = 1000
    early_stopping = 1e9
    repeat = 3

    learn_start = [25, 100, 500]
    target_update_frequency = [50, 200, 500]
    train_frequency = [1, 4, 16]
    discount = [0.8, 0.95, 0.99]
    epsilon_decay = [0.9, 0.99, 0.995]
    agent_batch_size = [16, 32, 64]
    hidden_sizes = [(128, 128), (512, 128), (256, 64)]
    learning_rate = [1e-3, 5e-3, 1e-4, 5e-4]

    trial_space = [learn_start,
                   target_update_frequency,
                   train_frequency,
                   discount,
                   epsilon_decay,
                   agent_batch_size,
                   hidden_sizes,
                   learning_rate]
    trials = _generate_trials(trial_space, r_factor=5)

    for i, (ls, tuf, tf, d, ed, abs, hs, lr) in enumerate(trials):
        for j in range(repeat):
            name = save_name + f"_gadqn_trial={i}_repeat={j}"
            _test_op_with_dqnga(save_path, name, workload, n_trial, early_stopping,
                                ls, tuf, tf, d, ed, abs, hs, lr, RewardFunction.R3)


def trial_ga(save_path, save_name, workload, trials=10):
    """
    Run a number of experiments using GA tuner.

    Experiment Name
    ---------------
    trial_ga
    """
    n_trial = 2000
    early_stopping = 1e9

    for i in range(trials):
        name = save_name + "_ga_trial=" + str(i)
        _test_op_with_ga(save_path, name, workload, n_trial, early_stopping)


def trial_gadqn(save_path, save_name, workload_name, trials=10, reward_function=RewardFunction.R3):
    """
    Run a number of experiments using GA-DQN tuner.

    Experiment Name
    ---------------
    trial_gadqn
    """

    # defaults
    n_trial = 2000
    early_stopping = 1e9
    learn_start = 100
    update_frequency = 200
    train_frequency = 4
    discount = 0.95
    epsilon_decay = 0.99
    agent_batch_size = 64
    hidden_size = (256, 64)
    learning_rate = 1e-3

    for i in range(trials):
        name = save_name + "_gadqn_trial=" + str(i)
        _test_op_with_dqnga(save_path, name, workload_name, n_trial, early_stopping,
                                     learn_start, update_frequency, train_frequency,
                                     discount, epsilon_decay, agent_batch_size,
                                     hidden_size, learning_rate, reward_function)


def compare_gadqn_with_ga(save_path, save_name, workload, expected_trials, prev_results_dir=None):
    """
    Compare iterations of ga with iterations of its DQN counterpart.
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

    # Create new graph displaying averages of both plots
    comparison_plot(save_path,
                    "best_score_comparison",
                    f"Computational performance of {workload['name']} as hardware measurements\nincrease on GA-DQN compared to GA",
                    "Trial number (Hardware measurements)",
                    "Best GigaFLOPS - Higher is better",
                    gadqn_tuning,
                    ga_tuning,
                    gadqn_steps,
                    ga_steps)


def compare_reward_with_ga(save_path, save_name, workload, expected_trials, prev_results_dir=None):
    """
    Compare iterations of ga with iterations of its DQN counterpart.
    Average and log these results in a graph.

    Experiment Name
    ---------------
    compare_reward
    """
    gadqn_tuning = []
    ga_tuning = []
    gadqn_steps = None
    ga_steps = None

    # collect best score results
    for reward_function in RewardFunction:
        reward_tuning = []
        for i in range(expected_trials):
            gadqn_path = save_path + save_name + f"_reward={reward_function}_gadqn_trial=" + str(i)
            y_data = DynamicPlot.load(gadqn_path, "best_score").y_data
            if not gadqn_steps:
                gadqn_steps = DynamicPlot.load(gadqn_path, "best_score").x_data
            reward_tuning.append(y_data)
        gadqn_tuning.append(reward_tuning)

    for i in range(expected_trials):
        ga_path = prev_results_dir if prev_results_dir else save_path + save_name
        ga_path = ga_path + "_ga_trial=" + str(i)
        y_data = DynamicPlot.load(ga_path, "best_score").y_data
        ga_tuning.append(y_data)
        if not ga_steps:
            ga_steps = DynamicPlot.load(ga_path, "best_score").x_data

    # Create new graph displaying averages of both plots
    reward_comparison_plot(save_path,
                           "best_score_comparison",
                           f"Computational performance of {workload['name']} as hardware measurements\nincrease on varying reward functions.",
                           "Trial number (Hardware measurements)",
                           "Best GigaFLOPS - Higher is better",
                           gadqn_tuning,
                           ga_tuning,
                           gadqn_steps,
                           ga_steps)
