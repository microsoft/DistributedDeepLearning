define PROJECT_HELP_MSG
Usage:
    make help                   show this message
endef
export PROJECT_HELP_MSG

DATA_DIR:=/mnt/imagenet
PWD:=$(shell pwd)
#PROJ_ROOT:=$(shell dirname $(PWD))
setup_volumes:=-v $(PWD)/src/execution:/mnt/script \
	-v $(DATA_DIR):/mnt/input \
	-v $(DATA_DIR)/temp/model:/mnt/model \
	-v $(DATA_DIR)/temp/output:/mnt/output


setup_environment:=--env AZ_BATCHAI_INPUT_TRAIN='/mnt/input/train' \
	--env AZ_BATCHAI_INPUT_TEST='/mnt/input/test' \
	--env AZ_BATCHAI_OUTPUT_MODEL='/mnt/model' \
	--env AZ_BATCHAI_JOB_TEMP_DIR='/mnt/output'

name_prefix:=masalvar

define serve_notebbook
 nvidia-docker run -it \
 $(setup_volumes) \
 $(setup_environment) \
 -p 10000:10000 \
 $(1) bash -c "jupyter notebook --port=10000 --ip=* --allow-root --no-browser --notebook-dir=/mnt/script"
endef

define execute_mpi
 nvidia-docker run -it \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --privileged \
 $(1) bash -c "mpirun -np 2 -H localhost:2 python /mnt/script/ImagenetKerasHorovod.py"
endef

define execute_mpi_intel
 nvidia-docker run -it \
 $(setup_volumes) \
 $(setup_environment) \
 --env DISTRIBUTED='True' \
 --privileged \
 $(1) bash -c " source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh; mpirun -n 2 -host localhost -ppn 2 -env I_MPI_DAPL_PROVIDER=ofa-v2-ib0 -env I_MPI_DYNAMIC_CONNECTION=0 python /mnt/script/ImagenetKerasHorovod.py"
endef

define execute
 nvidia-docker run -it \
 $(setup_volumes) \
 $(setup_environment) \
 $(1) bash -c "python /mnt/script/ImagenetKerasHorovod.py"
endef

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(name_prefix)/horovod-keras Docker/horovod

build-intel:
	docker build -t $(name_prefix)/horovod-intel-keras Docker/horovod-intel

notebook:
	$(call serve_notebbook, $(name_prefix)/horovod-keras)

run-mpi:
	$(call execute_mpi, $(name_prefix)/horovod-keras)

run-mpi-intel:
	$(call execute_mpi_intel, $(name_prefix)/horovod-intel-keras)

run:
	$(call execute, $(name_prefix)/horovod-keras)

push:
	docker push $(name_prefix)/horovod-keras

push-intel:
	docker push $(name_prefix)/horovod-intel-keras


.PHONY: help build push
