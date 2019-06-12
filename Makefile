# This makefile is used to test the cookiecutter
# To use this you will need to create a .dev_env file and add the subscription_id to it
include .dev_env

cookiecutter:
ifdef subscription_id
	cd ../ && cookiecutter DistributedDeepLearning --no-input \
	 subscription_id=${subscription_id} \
	 resource_group=mstestdistrg \
	 data=/mnt/imagenet_test \
	 vm_size=Standard_NC24rs_v3 \
	 project_name=mstestdist \
	 image_name=mstestdist
else
	@echo "You need to create a .dev_env file with subscription_id in it"
endif

clean: 
	rm -rf ../mstestdist