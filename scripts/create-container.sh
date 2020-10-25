#!/bin/bash

docker run \
    -v="$(pwd)":/workspace \
    -w=/workspace \
    -it \
    benchmark-tvm:1.0 bash
