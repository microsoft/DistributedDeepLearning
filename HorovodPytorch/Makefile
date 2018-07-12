define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  make Horovod PyTorch image with Open MPI
    make build-intel            make Horovod PyTorch image with Intel MPI
    make run-mpi				run training using Open MPI image
    make run-mpi-intel			run training using Intel MPI image
    make run					run training in non-distributed mode
    make push					push Horovod PyTorch image with Open MPI
    make push-intel				push Horovod PyTorch image with Intel MPI
endef
export PROJECT_HELP_MSG

ID:=ddl
LOCATION:=eastus
GROUP_NAME:=batch${ID}rg
STORAGE_ACCOUNT_NAME:=batch${ID}st
CONTAINER_NAME:=batch${ID}container
FILE_SHARE_NAME:=batch${ID}share
VM_SIZE:=Standard_NC24rs_v3
NUM_NODES:=8
CLUSTER_NAME:=msv100
JOB_NAME:=horovod_pytorch
SELECTED_SUBSCRIPTION:="Team Danielle Internal"
WORKSPACE:=workspace
GPU_TYPE:=V100
EXPERIMENT:=experiment_${GPU_TYPE}
PROCESSES_PER_NODE:=4
NFS_NAME:="batch${ID}nfs"
FAKE_DATA_LENGTH:=1281167

name_prefix:=masalvar
tag:=9-1.8-.13.2 # Cuda - TF version - Horovod version
intel-image:=$(name_prefix)/horovod-pytorch-intel
open-image:=$(name_prefix)/horovod-pytorch
script:=\$$AZ_BATCHAI_INPUT_SCRIPTS/ImagenetPytorchHorovod.py

DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
setup_volumes:=-v $(PWD)/src/execution:/mnt/script \
	-v $(DATA_DIR):/mnt/input \
	-v $(DATA_DIR)/temp/model:/mnt/model \
	-v $(DATA_DIR)/temp/output:/mnt/output


setup_environment:=--env AZ_BATCHAI_INPUT_TRAIN='/mnt/input' \
	--env AZ_BATCHAI_INPUT_TEST='/mnt/input' \
	--env AZ_BATCHAI_OUTPUT_MODEL='/mnt/model' \
	--env AZ_BATCHAI_JOB_TEMP_DIR='/mnt/output'


define execute_mpi
 nvidia-docker run --shm-size 8G -it \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --env FAKE=$(FAKE) \
 --env FAKE_DATA_LENGTH=$(FAKE_DATA_LENGTH) \
 --privileged \
 $(1) bash -c "mpirun -np 2 -H localhost:2 python /mnt/script/ImagenetPytorchHorovod.py"
endef


define execute_mpi_intel
 nvidia-docker run --shm-size 8G -it \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --env FAKE=$(FAKE) \
 --env FAKE_DATA_LENGTH=$(FAKE_DATA_LENGTH) \
 --privileged \
 $(1) bash -c " source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh; mpirun -n 2 -host localhost -ppn 2 -env I_MPI_DAPL_PROVIDER=ofa-v2-ib0 -env I_MPI_DYNAMIC_CONNECTION=0 python /mnt/script/ImagenetPytorchHorovod.py"
endef

define execute
 nvidia-docker run --shm-size 8G -it \
 $(setup_volumes) \
 $(setup_environment) \
 $(1) bash -c "python /mnt/script/ImagenetPytorchHorovod.py"
endef

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(name_prefix)/horovod-pytorch Docker/horovod

build-intel:
	docker build -t $(name_prefix)/horovod-intel-pytorch Docker/horovod-intel

run-mpi:
	$(call execute_mpi, $(name_prefix)/horovod-pytorch)

run-mpi-intel:
	$(call execute_mpi_intel, $(name_prefix)/horovod-intel-pytorch)

run:
	$(call execute, $(name_prefix)/horovod-pytorch)

push:
	docker push $(name_prefix)/horovod-pytorch

push-intel:
	docker push $(name_prefix)/horovod-intel-pytorch

include ../include/*.mk

upload-scripts: set-storage
	$(call upload_script, src/execution/ImagenetPytorchHorovod.py)
	$(call upload_script, src/execution/timer.py)

.PHONY: help build push
