"""
Test implementation of the tuner.
"""

import tvm
from tvm import relay
from tvm import autotvm
from tvm.relay import testing
from .ga_dqn_tuner import DQNGATuner

target = tvm.target.Target("cuda")


def _get_relay_convolution():
    """
    Create simple relay convolution.
    """
    dtype = "float32"
    shape = (1, 3, 8, 8)
    data = relay.var("data", shape=shape, dtype=dtype)
    weight = relay.var("weight")
    out = relay.nn.conv2d(data, weight, channels=16, kernel_size=(3, 3), padding=(1, 1))
    net = relay.Function(relay.analysis.free_vars(out), out)
    return testing.create_workload(net)


def test_convolution(n_trial=2000,
                     early_stopping=400,
                     learn_start=50,
                     memory_capacity=1000,
                     update_frequency=50,
                     discount=0.99,
                     epsilon=(1.0, 0.01, 0.99)):
    """
    Test simple convolution with RLTuner.
    """
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
                         update_frequency=update_frequency,
                         discount=discount,
                         epsilon=epsilon,
                         debug=True)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)])


if __name__ == "__main__":
    test_convolution()
