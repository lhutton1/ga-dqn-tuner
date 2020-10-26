#!/bin/bash

# TVM dependencies
apt-get update
apt-get install -y --no-install-recommends \
    gcc \
    libtinfo-dev \
    zlib1g-dev \
    build-essential \
    libedit-dev \
    libxml2-dev \
    libssl-dev

# Clone TVM
git clone --recursive https://github.com/apache/incubator-tvm.git tvm
cd tvm
git checkout bb4179e2867a1d3cb9bd56c707681cc66c07d459 # pin TVM version
git submodule init
git submodule update

# Config TVM
mkdir build
cd build
cat > config.cmake <<EOF
set(USE_RPC ON)
set(USE_GRAPH_RUNTIME ON)
set(USE_LLVM ON)
set(USE_RANDOM ON)
set(USE_SORT ON)
EOF

# Build TVM
cmake ..
make -j4

# Install python dependencies
cd ../python
pip3 install --upgrade pip setuptools
pip3 install -e .[extra_feature]
pip3 install torch==1.4.0
