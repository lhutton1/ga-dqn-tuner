import tvm
from tvm import relay
from tvm.relay import testing
from tvm import autotvm
from tvm.autotvm.tuner.rl_optimizer.rl_tuner import DQNTuner

target = tvm.target.Target("cuda")


def get_relay_convolution():
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


def test_convolution(save_path,
                     save_name,
                     n_trial,
                     early_stopping,
                     learn_start,
                     memory_capacity,
                     update_frequency,
                     discount,
                     epsilon_max,
                     epsilon_min,
                     epsilon_decay):
    """
    Test a simple convolution using RLTuner.
    """
    mod, params = get_relay_convolution()
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=tvm.target.Target("llvm"),
        params=params)
    runner = autotvm.LocalRunner(number=1, repeat=4)
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(build_func="default", n_parallel=1),
                                            runner=runner)
    prefix = f"[Task 1/1]"
    tuner_obj = DQNTuner(tasks[0],
                         learn_start=learn_start,
                         memory_capacity=memory_capacity,
                         update_frequency=update_frequency,
                         discount=discount,
                         epsilon=(epsilon_max, epsilon_min, epsilon_decay),
                         debug=True)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(100, prefix=prefix)])
    tuner_obj.save_model(save_path, save_name)


def trial_parameters(json_config):
    """
    Run a series of tuning trials with varying hyperparameters. Log results.
    """
    config = json_config["trial_hyperparameters"]
    save_path = config["save_path"]
    save_name = config["save_name"]

    n_trial = 4000
    early_stopping = 1e9
    learn_start = 500
    memory_capacity = 1000
    update_frequency = 50
    discount = 0.99
    epsilon = (1.0, 0.01, 0.99)

    # replay memory trails
    for ls, mc in [(50, 200), (50, 50), (200, 1000), (500, 1000), (1000, 1000)]:
        save_name = save_name + "_learn_start=" + str(learn_start) + "_memory_capacity=" + str(memory_capacity)
        test_convolution(save_path, save_name, n_trial, early_stopping,
                         ls, mc, update_frequency,
                         discount, epsilon[0], epsilon[1], epsilon[2])

    # update frequency trials
    for uf in [1, 10, 25, 50, 200]:
        save_name = save_name + "_update_frequency=" + str(update_frequency)
        test_convolution(save_path, save_name, n_trial, early_stopping,
                         learn_start, memory_capacity, uf,
                         discount, epsilon[0], epsilon[1], epsilon[2])

    # discount trials
    for d in [0.99, 0.95, 0.85, 0.75]:
        save_name = save_name + "_discount=" + str(discount)
        test_convolution(save_path, save_name, n_trial, early_stopping,
                         learn_start, memory_capacity, update_frequency,
                         d, epsilon[0], epsilon[1], epsilon[2])
