Bootstrap: docker
From: ubuntu:18.04

%files
./* /benchmark-tvm

%post

# Set correct locale
apt-get -y update && apt-get install -y locales
locale-gen en_GB.UTF-8
export LANG=en_GB.UTF-8
export LANGUAGE=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8

# Install basics
apt-get install --no-install-recommends -y \
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

mkdir /install
cd /install
cp /benchmark-tvm/singularity-utils/* /install/
bash ./install-cmake.sh
bash ./install-llvm.sh
bash ./install-tvm.sh

%runscript