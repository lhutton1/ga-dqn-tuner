# Generating high-performance code for deep learning workloads: a reinforcement learning based approach.

This project aims to apply reinforcement learning to auto-tuning in AutoTVM (part of the TVM machine learning compiler),
in order to improve the experience of the end user. Currently, reinforcement learning is applied to the GATuner - a genetic algorithm
that repeatedly applies elitism, 2-point crossover and mutation to a population. Named **GA-DQN**, the new tuner uses two independent 
deep Q-network (DQN)'s that are applied to crossover and mutation. Crossover is completed by allowing DQN to suggest the point at 
which to crossover a gene, while, mutation is completed by allowing DQN to select which detail to randomly mutate. In addition, an evaluation 
framework is provided to assess the performance of GA-DQN.

![GA-DQN tuning pipeline](/assets/ga-dqn-pipeline.png "GA-DQN tuning pipeline")

## Usage
To use the tuner, TVM must be installed and visible within your python environment. Due to needing additional features not available in a released 
version of TVM, a forked version of TVM is used which applies a small amount debugging code and a fix to the PyTorch front-end parser. A pinned
version is also used as TVM is mostly in a development stage and the API's used are unstable. Consequently, the GA-DQN tuner has only been tested
with this specific commit, along with small modifications ontop. The required version can be pulled from git like so:

```bash
git clone --recursive https://github.com/lhutton1/tvm.git tvm
cd tvm
git checkout autotvm-measure-remote-time
git checkout d2452502b9486a7993d9dec3d04e449efdd81cf7
```

TVM also requires a number of dependencies such as: Cuda, Python3.6, LLVM, XGBoost (for the XGBTuner) and PyTorch (for the GA-DQN tuner). As such, we recommend using a containerised environment powered by Singularity. Similar to docker, an image must be built from which containers can be run based on the image. First install Singularity, then build the image using a simple script provided:

```bash
# Install Singularity
sudo wget -O- http://neuro.debian.net/lists/xenial.us-ca.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    sudo apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9 && \
    sudo apt-get update
    
sudo apt-get install -y singularity-container
    

# Build image
./create_image.sh
```

From this a container can be created and GA-DQN can be run from within this container using the presented shell:
```bash
./create_container.sh rl-tuner.simg
```

Now in the shell, test your container works correctly by attempting to run the evaluation framework help prompt:
```bash
python driver.py --help
```

_Note: This has been tested on a Ubuntu 18.04 setup and is not guaranteed to work with other operating systems. These scripts have also been tested on the University of Leeds HPC cluster, ARC._

_Note: it is possible to build TVM and install its dependencies from scratch, although this is not recommended due to the number of packages required. The process required should be similar to that provided in `create_image.sh` script. However, it is recommended you create a new virtual environment for python in this process._

## RL Tuner
GA-DQN is a tuner that combines advancements in reinforcement learning and the genetic algorithm tuner that currently exists in TVM. Two independent deep Q-network (DQN)'s are used to suggest where to crossover genes and which detail of a gene to mutate.

## GA Tuner
The GA tuner is code obtained from the open source TVM compiler. It is here for convenience and to allow a small amount of debug code to be added so that it can be evaluated. This work is not my own.

## Evaluation framework (tools)
Provides a series of tools and experiments to quickly test various tuning algorithms in AutoTVM. Use tune and benchmark commands on a series of pre-trained models to evaluate random, genetic algorithm, extreme gradient boost and GA-DQN algorithms. Use the experiment framework to evaluate various aspects of GA-DQN, with graphical monitoring.

A command line driver is provided for this framework:
```bash
python driver.py -m=tune -c=../config-example.json
python driver.py -m=benchmark -c=../config-example.json
python driver.py -m=experiment -c=../config-example.json
```

