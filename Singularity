Bootstrap: docker
From: nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04


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

# Install to install dir
mkdir -p /install
cd /install

# Install cmake
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc |
    sudo apt-key add -
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
apt-get update

apt-get -y --no-install-recommends install cmake

# Install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 11
rm -f llvm.sh

# Install TVM Dependencies
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
set(USE_GRAPH_RUNTIME_DEBUG ON)
set(USE_LLVM ON)
set(USE_RANDOM ON)
set(USE_SORT ON)
set(USE_CUDA ON)
EOF

# Build TVM
cmake ..
make -j4

# Install python dependencies
cd ../python
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .[extra_feature]
python3 -m pip install -e .[test]
python3 -m pip install packaging
python3 -m pip install torch>=1.4.0
python3 -m pip install torchvision>=0.5.0

# Cuda paths
export PATH=/usr/local/nvidia/bin:${PATH}
export PATH=/usr/local/cuda/bin:${PATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
export C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
export LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compact:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compact:${LD_LIBRARY_PATH}
export CUDA_HOME="/usr/local/cuda"


%runscript
