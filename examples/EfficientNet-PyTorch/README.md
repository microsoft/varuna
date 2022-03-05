# EfficientNet PyTorch


Taken from https://github.com/lukemelas/EfficientNet-PyTorch

This very simple example is to test the installation and basic funtionalities of Varuna.

This example contains an implementation of [EfficientNet](https://arxiv.org/abs/1905.11946), evaluated on ImageNet data. It is adapted from the standard EfficientNet-PyTorch imagenet script. 

To run on Imagenet, place your `train` and `val` directories in `data`. 

A small version of imagenet can be obtained at https://github.com/fastai/imagenette

## Setup

To install the efficient example:
```bash
cd <varuna-folder>
cd examples/EfficientNet-PyTorch/
pip install -e .
```

## Usage


Running on CPU on a single machine:

```bash
 python3 -m varuna.run_varuna --machine_list list --no_morphing --gpus_per_node 1 --batch_size 2 --nstages 1 --chunk_size 1 --code_dir . main.py data -e -a 'efficientnet-b0' --pretrained --varuna --lr 0.001 --epochs 1 --cpu

```

Running on 1 GPU on a single machine:

```bash
 python3 -m varuna.run_varuna --machine_list list --no_morphing --gpus_per_node 1 --batch_size 2 --nstages 1 --chunk_size 1 --code_dir . main.py data -e -a 'efficientnet-b0' --pretrained --varuna --lr 0.001 --epochs 1

```

In each case, the model should train for one epoch (hard-coded to 10 steps) and then run validation.

Output and error logs are created in the folder `ssh_logs/`
