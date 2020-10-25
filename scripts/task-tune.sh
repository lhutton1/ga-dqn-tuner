#!/bin/bash

wget -N https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx

tvmc tune \
  --target="llvm" \
  --model-format=onnx \
  --output=autotuner_records.json \
  resnet50-v2-7.onnx