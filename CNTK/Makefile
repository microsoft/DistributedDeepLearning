DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
image-open:=hoaphumanoid/CNTK:distributed
open-path:=$(PWD)/Docker
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py
include ../include/build.mk

