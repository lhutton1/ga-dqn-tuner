import json
import os
import time

import tvm
from tvm import relay
from tvm import autotvm
from tvm import rpc
from tvm.contrib import util
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
import numpy as np
import torch

from .get_model import get_model


def get_input_info(graph, params):
    """
    Get shape and dtype of the input tensors for a compiled model.
    """
    shape_dict = {}
    dtype_dict = {}
    graph = json.loads(graph)
    param_names = params.keys()
    for node_id in graph["arg_nodes"]:
        node = graph["nodes"][node_id]
        name = node["name"]
        if name not in param_names:
            shape_dict[name] = graph["attrs"]["shape"][1][node_id]
            dtype_dict[name] = graph["attrs"]["dltype"][1][node_id]
    return shape_dict, dtype_dict


def generate_tensor_data(shape, dtype):
    """
    Generate data to produce a tensor of given shape and dtype for TVM.
    """
    if "int8" in dtype:
        tensor = np.random.randint(128, size=shape, dtype=dtype)
    else:
        tensor = np.random.uniform(-1, 1, size=shape).astype(dtype)
    return tensor


def generate_pytorch_data(input_shapes, target_string, dtype):
    """
    Generate data to produce a tensor of given shape and dtype for PyTorch.
    """
    data = []
    for input_shape in input_shapes:
        if target_string == "cuda":
            if dtype == "int":
                data.append(torch.randint(0,10, input_shape[1], device="cuda"))
            else:
                data.append(torch.cuda.FloatTensor(input_shape[1]).normal_())
        else:
            data.append(torch.randn(input_shape[1]))
    return data


def make_inputs_dict(shape_dict, dtype_dict):
    """
    Make the inputs dictionary for a graph. Here, random input tensors
    for each input are produced.
    """
    inputs_dict = {}
    for input_name in shape_dict:
        shape = shape_dict[input_name]
        dtype = dtype_dict[input_name]
        data = generate_tensor_data(shape, dtype)
        inputs_dict[input_name] = data
    return inputs_dict


def extract_profile_data(times):
    """
    Provided a series of execution times calculate the mean, std, max and min.
    """
    mean_ts = np.mean(times)
    std_ts = np.std(times)
    max_ts = np.max(times)
    min_ts = np.min(times)

    header = "Execution time summary:\n{0:^10} {1:^10} {2:^10} {3:^10}".format(
        "mean (s)", "max (s)", "min (s)", "std (s)")
    stats = "{0:^10.5f} {1:^10.5f} {2:^10.5f} {3:^10.5f}".format(
        mean_ts, max_ts, min_ts, std_ts)

    return header, stats


def compile_model(mod, params, target_string, tuning_records=None, model_name=""):
    """
    Compile TVM model. Apply tuning records if they exist.
    """
    target = tvm.target.Target(target_string)
    target_host = "llvm"

    if tuning_records:
        if model_name:
            model_str = "_model=" + model_name + ".json"
            tuning_records = tuning_records[:-5]
            tuning_records += model_str
        if os.path.exists(tuning_records):
            print('applying tuning history...')
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3):
                graph_module = relay.build(mod, target, params=params, target_host=target_host)
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph_module = relay.build(mod, target, params=params, target_host=target_host)

    return graph_module.get_json(), graph_module.get_lib(), graph_module.get_params()


def run_model_tvm(graph, lib, params, run_settings, model_name, tuning_records=None):
    """
    Run TVM model. Apply tuning records if they exist.
    """
    profile = run_settings['profile']
    device = run_settings['device']
    repeat = run_settings['repeat']

    session = rpc.LocalSession()
    ctx = session.cpu() if device == "cpu" else session.gpu()
    is_tuned = True if tuning_records else False

    lib_name = "mod.so"
    temp = util.tempdir()
    lib_path = temp.relpath(lib_name)
    lib.export_library(lib_path)
    session.upload(lib_path)
    lib = session.load_module(lib_name)

    if profile:
        module = debug_runtime.create(graph, lib, ctx, dump_root=f"results/prof_{model_name}_tuned={is_tuned}")
    else:
        module = runtime.create(graph, lib, ctx)

    saved_params = relay.save_param_dict(params)
    module.load_params(saved_params)

    shape_dict, dtype_dict = get_input_info(graph, params)
    inputs_dict = make_inputs_dict(shape_dict, dtype_dict)
    module.set_input(**inputs_dict)

    if profile:
        module.run()

    timer = module.module.time_evaluator("run", ctx, 1, repeat=repeat)
    prof_result = timer()
    times = prof_result.results
    header, stats = extract_profile_data(times)

    filename = f'results/stat_table_{model_name}_tuned={is_tuned}'
    with open(filename, 'w') as f:
        print("%s\n%s\n" % (header, stats), filename, file=f)
    print("%s\n%s\n" % (header, stats))


def run_model_pytorch(trace, input_shapes, run_settings, model_name, target_string):
    """
    Run model using the standard pytorch runtime.
    """
    repeat = run_settings['repeat']

    # TODO extract datatype required from PyTorch model itself
    dtype = "int" if model_name == "bert" else "float"
    generated_data = generate_pytorch_data(input_shapes, target_string, dtype)

    # First run is always slower due to on-the-fly optimization, skip it.
    skip_first_run = True
    times = []
    for _ in range(repeat + 1):
        start = time.time()
        # Note: only a single input is currently supported
        trace(*generated_data)
        end = time.time() - start

        if skip_first_run:
            skip_first_run = False
            continue
        times.append(end)

    header, stats = extract_profile_data(times)

    filename = f'results/stat_table_{model_name}_pytorch'
    with open(filename, 'w') as f:
        print("%s\n%s\n" % (header, stats), filename, file=f)
    print("%s\n%s\n" % (header, stats))


def benchmark_models(data):
    """
    Compile and run a TVM model and benchmark it. Then run the same model using PyTorch
    and display statistics comparing the two.
    """
    target_string = data['target']
    tuning_records = data['autotuner_settings'].get('tuning_records') or ""
    run_settings = data['run_settings']

    for model in data['models']:
        for executor in ["tvm", "pytorch"]:
            trace, input_shapes = get_model(model['name'], model['type'])

            if executor == "tvm":
                mod, params = relay.frontend.from_pytorch(trace, input_shapes)

                print(f"Compiling TVM model {model['name']}")
                graph, lib, params = compile_model(mod, params, target_string, tuning_records, model['name'])
                
                print(f"Running TVM model {model['name']}")
                run_model_tvm(graph, lib, params, run_settings, model['name'], tuning_records)
            elif executor == "pytorch":
                run_model_pytorch(trace, input_shapes, run_settings, model['name'], target_string)
            else:
                raise ValueError(f"executor {executor} not supported.")
