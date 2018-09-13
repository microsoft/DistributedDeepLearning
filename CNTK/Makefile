DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
FAKE:='False'
FAKE_DATA_LENGTH:=1281167
image-open:=hoaphumanoid/cntk:distributed-openmpi3
open-path:=$(PWD)/Docker/cntk
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_cntk.py
include ../include/build.mk

