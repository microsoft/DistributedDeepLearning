DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
name_prefix:=masalvar
tag:=9-1.8-.13.2 # Cuda - TF version - Horovod version
image-intel:=$(name_prefix)/horovod-intel-keras:$(tag)
intel-path:=$(PWD)/Docker/horovod-intel
image-open:=microsoft/cntk:2.5.1-gpu-python3.5-cuda9.0-cudnn7.0
open-path:=$(PWD)/Docker/horovod
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py
include ../include/build.mk

