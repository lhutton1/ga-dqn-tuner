#!/usr/bin/env python3
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

from get_model import get_model


def get_input_info(graph, params):
    """Return the 'shape' and 'dtype' dictionaries for the input
    tensors of a compiled module.

    .. note::
        We can't simply get the input tensors from a TVM graph
        because weight tensors are treated equivalently. Therefore, to
        find the input tensors we look at the 'arg_nodes' in the graph
        (which are either weights or inputs) and check which ones don't
        appear in the params (where the weights are stored). These nodes
        are therefore inferred to be input tensors.

    Parameters
    ----------
    graph_str : str
        JSON graph of the module serialized as a string.
    params : bytearray
        Params serialized as a bytearray.

    Returns
    -------
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    """
    shape_dict = {}
    dtype_dict = {}
    graph = json.loads(graph)
    param_names = params.keys()
    for node_id in graph["arg_nodes"]:
        node = graph["nodes"][node_id]
        # If a node is not in the params, infer it to be an input node
        name = node["name"]
        if name not in param_names:
            shape_dict[name] = graph["attrs"]["shape"][1][node_id]
            dtype_dict[name] = graph["attrs"]["dltype"][1][node_id]

    return shape_dict, dtype_dict


def generate_tensor_data(shape, dtype, fill_mode):
    """Generate data to produce a tensor of given shape and dtype.

    Random data generation depends on the dtype. For int8 types,
    random integers in the range 0->255 are generated. For all other
    types, random floats are generated in the range -1->1 and then
    cast to the appropriate dtype.

    This is used to quickly generate some data to input the models, as
    a way to check that compiled module is sane for running.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor.
    dtype : str
        The dtype of the tensor.
    fill_mode : str
        The fill-mode to use, either "zeros", "ones" or "random".

    Returns
    -------
    tensor : np.array
        The generated tensor as a np.array.
    """
    if fill_mode == "zeros":
        tensor = np.zeros(shape=shape, dtype=dtype)
    elif fill_mode == "ones":
        tensor = np.ones(shape=shape, dtype=dtype)
    elif fill_mode == "random":
        if "int8" in dtype:
            tensor = np.random.randint(128, size=shape, dtype=dtype)
        else:
            tensor = np.random.uniform(-1, 1, size=shape).astype(dtype)
    else:
        raise ValueError("unknown fill-mode: {}".format(fill_mode))

    return tensor


def generate_pytorch_data(input_shapes, fill_mode):
    """ Create pytorch data tensors. """
    data = []
    for input_shape in input_shapes:
        if fill_mode == "random":
            data.append(torch.randn(input_shape))
        else:
            raise ValueError(f"fill mode {fill_mode} not supported.")


def make_inputs_dict(shape_dict, dtype_dict, fill_mode):
    """Make the inputs dictionary for a graph.

    Use data from 'inputs' where specified. For input tensors
    where no data has been given, generate data according to the
    chosen fill-mode.

    Parameters
    ----------
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    fill_mode : str
        The fill-mode to use when generating tensor data.
        Can be either "zeros", "ones" or "random".

    Returns
    -------
    inputs_dict : dict
        Complete inputs dictionary - {input_name: np.array}.
    """
    inputs = {}

    # First check all the keys in inputs exist in the graph
    for input_name in inputs:
        if input_name not in shape_dict.keys():
            raise ValueError(
                "the input tensor '{}' is not in the graph. Expected inputs: '{}'".format(
                    input_name, shape_dict.keys()
                )
            )

    # Now construct the input dict, generating tensors where no
    # data already exists in 'inputs'
    inputs_dict = {}
    for input_name in shape_dict:
        if input_name in inputs.keys():
            inputs_dict[input_name] = inputs[input_name]
        else:
            shape = shape_dict[input_name]
            dtype = dtype_dict[input_name]
            data = generate_tensor_data(shape, dtype, fill_mode)
            inputs_dict[input_name] = data

    return inputs_dict

def extract_profile_data(times):
    """Provided a series of execution times calculate the mean, std, max and min. """
    mean_ts = np.mean(times)
    std_ts = np.std(times)
    max_ts = np.max(times)
    min_ts = np.min(times)

    header = "Execution time summary:\n{0:^10} {1:^10} {2:^10} {3:^10}".format(
        "mean (s)", "max (s)", "min (s)", "std (s)"
    )
    stats = "{0:^10.5f} {1:^10.5f} {2:^10.5f} {3:^10.5f}".format(mean_ts, max_ts, min_ts, std_ts)

    return header, stats


def compile_model(mod, params, target_string, tuning_records=None):
    """ Compile TVM model. Apply tuning records if they exist. """
    target = tvm.target.Target(target_string)
    target_host = "llvm"

    if tuning_records and os.path.exists(tuning_records):
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3):
                graph_module = relay.build(mod, target, params=params, target_host=target_host)
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph_module = relay.build(mod, target, params=params, target_host=target)

    return graph_module.get_json(), graph_module.get_lib(), graph_module.get_params()


def run_model_tvm(graph, lib, params, run_settings, model_name, tuning_records=None):
    """ Run TVM model. Apply tuning records if they exist. """
    profile = run_settings['profile']
    device = run_settings['device']
    fill_mode = run_settings['fill_mode']
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
    inputs_dict = make_inputs_dict(shape_dict, dtype_dict, fill_mode)
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


def run_model_pytorch(trace, input_shapes, run_settings, model_name):
    """ Run model using the standard pytorch runtime. """
    profile = run_settings['profile']
    device = run_settings['device']
    fill_mode = run_settings['fill_mode']
    repeat = run_settings['repeat']

    generated_data = generate_pytorch_data(input_shapes, fill_mode)
    # first run is always slower due to on the fly optimization, skip it.
    skip_first_run = True
    times = []
    for run in range(repeat + 1):
        start = time.time()*1000
        model(generated_data)
        if skip_first_run:
            times.append(time.time()*1000 - start)
            skip_first_run = False

    header, stats = extract_profile_data(times)

    filename = f'result/stat_table_{model_name}_pytorch'
    with open(filename, 'w') as f:
        print("%s\n%s\n" % (header, stats), filename, file=f)
    print("%s\n%s\n" % (header, stats))


if __name__ == '__main__':
    with open('config.json') as json_file:
        data = json.load(json_file)
        target_string = data['target']
        tuning_records = data['autotuner_settings']['tuning_records']
        run_settings = data['run_settings']

        for model in data['models']:
            for executor in ["tvm", "pytorch"]:
                trace, input_shapes = get_model(model['name'], model['type'])

                if executor == "tvm":
                    mod, params = relay.frontend.from_pytorch(trace, input_shapes)

                    print(f"Compiling TVM model {model['name']}")
                    graph, lib, params = compile_model(mod, params, target_string, tuning_records)
                    
                    print(f"Running TVM model {model['name']}")
                    run_model_tvm(graph, lib, params, run_settings, model['name'], tuning_records)
                else if executor == "pytorch":
                    run_model_pytorch(trace, input_shapes, run_settings, model['name'])
                else:
                    ValueError(f"executor {executor} not supported.")