FROM ubuntu:18.04
WORKDIR /install

RUN apt-get update --fix-missing

# Core
RUN apt-get install --no-install-recommends -y \
    ca-certificates \
    apt-transport-https \
    sudo \
    gnupg \
    lsb-release \
    software-properties-common \
    gpg-agent \
    wget \
    git \
    python3 \
    python3-dev \
    python3-setuptools \
    python3-pip

# Cmake=3.16.5
COPY docker/install-cmake.sh /install/install-cmake.sh
RUN bash /install/install-cmake.sh

# LLVM
COPY docker/install-llvm.sh /install/install-llvm.sh
RUN bash /install/install-llvm.sh

# TVM
COPY docker/install-tvm.sh /install/install-tvm.sh
RUN bash /install/install-tvm.sh