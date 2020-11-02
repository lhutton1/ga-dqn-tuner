#!/bin/bash

singularity shell --nv --containall --bind=/home/luke/benchmark-tvm/:/benchmark-tvm --pwd=/benchmark-tvm -H=/home/luke/benchmark-tvm/ benchmark-tvm.simg
