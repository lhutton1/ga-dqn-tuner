"""
A quick simple script to calculate the total configuration size of a model,
from params.json output.
"""
from pathlib import Path
import json

abs_path = Path("config_no" + "/example").resolve()

models = ["mobilenet_v2", "resnet18", "inception_v3", "bert", "simple_transformer"]
layers = [31, 16, 51, 6, 15]

for model, layer_count in zip(models, layers):
    count = 0
    for l in range(layer_count):
        with open(str(abs_path) + f"_model={model}_layer={l}/params.json") as f:
            dump = json.loads(f.read())
        count += dump['Space Size']
    print(f"Model = {model}, Config count = {count}")
