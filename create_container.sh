#!/bin/bash

# Designed to be executed from within the project root directory

image_path=$1

singularity shell \
  --nv \
  --containall \
  --bind=./:/benchmark-tvm \
  --pwd=/benchmark-tvm \
  -H=./ \
  "$image_path"
