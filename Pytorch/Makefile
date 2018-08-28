DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
name_prefix:=iliauk
tag:=latest
image-open:=$(name_prefix)/pytorch_gloo:$(tag)
open-path:=$(PWD)/Docker
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_gloo.py
include ../include/build.mk

