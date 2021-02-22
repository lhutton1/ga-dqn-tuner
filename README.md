# Applying reinforcement learning to auto-tuning in TVM

Provides an implemention of a tuning algorithm that leverages reinforcement learning techniques within the TVM framework.

## Tools
Provides a suite of tools and experiments for AutoTVM. 

## Run
A set of scripts for running the benchmarking tools on various hardware.

## Tuner
Reinforcement learning auto-tuner implmented into AutoTVM.

## Singularity
Provides a docker-like environment for testing out the implementation. Singularity is used as it is supported on ARC.

To build an image:
```
./create-image.sh
```

To start a container:
```
./create-container.sh
```

