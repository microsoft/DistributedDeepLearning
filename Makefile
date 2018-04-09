define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make cntk                   make cntk BAIT image
    make push-cntk              push cntk BAIT image to docker hub
endef
export PROJECT_HELP_MSG

DATA_DIR:=/mnt/
PWD:=$(shell pwd)
PROJ_ROOT:=$(shell dirname $(PWD))
setup_volumes:=-v $(PROJ_ROOT):/mnt/script \
	-v $(DATA_DIR):/mnt/input \
	-v $(DATA_DIR)/temp/model:/mnt/model \
	-v $(DATA_DIR)/temp/output:/mnt/output


setup_environment:=--env AZ_BATCHAI_INPUT_DATASET='/mnt/input' \
	--env AZ_BATCHAI_INPUT_SCRIPT='/mnt/script' \
	--env AZ_BATCHAI_OUTPUT_MODEL='/mnt/model' \
	--env AZ_BATCHAI_MOUNT_ROOT='/mnt/output'


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
 $(1) bash -c "mpirun -np 2 -H localhost:2 python /mnt/script/keras_mnist_advanced.py"
endef

help:
	echo "$$PROJECT_HELP_MSG" | less

build:
	docker build -t $(name_prefix)/horovod horovod

notebook:
	$(call serve_notebbook, $(name_prefix)/horovod)

run-mpi:
	$(call execute_mpi, $(name_prefix)/horovod)

push:
	docker push $(name_prefix)/horovod


.PHONY: help build push
