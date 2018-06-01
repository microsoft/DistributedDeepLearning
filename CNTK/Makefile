define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  make CNTK image with Open MPI
    make run-mpi				run training using Open MPI image
    make push					push CNTK image with Open MPI
endef
export PROJECT_HELP_MSG

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

name_prefix:=hoaphumanoid

define execute_mpi
 nvidia-docker run -it \
 $(setup_volumes) \
 $(setup_environment) \
 $(1) bash -c "mpirun -np 2 -H localhost:2 python /mnt/script/ImagenetEstimatorCNTK.py"
endef


help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(name_prefix)/cntk Docker

run-mpi:
	$(call execute_mpi, $(name_prefix)/cntk)

push:
	docker push $(name_prefix)/cntk

.PHONY: help build push
