define PROJECT_HELP_MSG
Makefile to control project aml_dist
Usage:
    help                        show this message
    build                       build docker image to use as control plane
	bash                        run bash inside runnin docker container
    stop                        stop running docker container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
PORT:=9999
TBOARD_PORT:=6006
NAME:=aml_dist # Name of running container

setup_environment_file:=--env-file .env
include .env

local_code_volume:=-v $(PWD):/workspace
volumes:=-v /tmp/azureml_runs:/tmp/azureml_runs \
				-v $(DATA):/data \
				-v ${HOME}/.bash_history:/root/.bash_history


help:
	echo "$$PROJECT_HELP_MSG" | less
	
build:
	docker build -t $(IMAGE_NAME) -f control/Docker/dockerfile control/Docker

/tmp/azureml_runs:
	mkdir -p /tmp/azureml_runs

run: /tmp/azureml_runs
	# Start docker running as daemon
	docker run $(local_code_volume) $(volumes) $(setup_environment_file)  \
	--name $(NAME) \
	-p $(PORT):$(PORT) \
	-p $(TBOARD_PORT):$(TBOARD_PORT) \
	-d \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-e HIST_FILE=/root/.bash_history \
	-it $(IMAGE_NAME) 

	# Attach to running container and create new tmux session
	docker exec -it $(NAME) bash -c "tmux new -s dist -n control"
	

bash:
	docker exec -it $(NAME) bash -c "tmux a -t dist"

stop:
	docker stop $(NAME)
	docker rm $(NAME)

.PHONY: help build run bash stop