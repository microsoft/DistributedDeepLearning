DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
name_prefix:=masalvar
tag:=9-1.8-.13.2 # Cuda - TF version - Horovod version
image-intel:=$(name_prefix)/horovod-intel-pytorch:$(tag)
intel-path:=$(PWD)/Docker/horovod-intel
image-open:=$(name_prefix)/horovod-pytorch:$(tag)
open-path:=$(PWD)/Docker/horovod

include ../include/build.mk

