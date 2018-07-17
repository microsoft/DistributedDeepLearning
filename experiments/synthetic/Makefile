include ../experiments_config.mk

FAKE_DATA_LENGTH:=1281167
EXPERIMENT:=experiment_synthetic_${GPU_TYPE}

include ../../include/control.mk

define submit_keras_intel
	$(call generate_job_intel,masalvar/horovod-intel-keras:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH} )
	$(call submit_job, $(3))
endef

define submit_keras
	$(call generate_job_openmpi,masalvar/horovod-keras:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_tf_intel
	$(call generate_job_intel,masalvar/horovod-intel:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_tf
	$(call generate_job_openmpi,masalvar/horovod:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_pytorch
	$(call generate_job_openmpi,masalvar/horovod-pytorch,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef


upload-scripts: set-storage
	$(call upload_script, ../../HorovodKeras/src/data_generator.py)
	$(call upload_script, ../../HorovodKeras/src/imagenet_keras_horovod.py)
	$(call upload_script, ../../HorovodTF/src/imagenet_estimator_tf_horovod.py)
	$(call upload_script, ../../HorovodPytorch/src/imagenet_pytorch_horovod.py)
	$(call upload_script, ../../common/timer.py)


create-cluster: upload-nodeprep-scripts
	az batchai cluster create \
	-w $(WORKSPACE) \
	--name ${CLUSTER_NAME} \
	--image UbuntuLTS \
	--vm-size ${VM_SIZE} \
	--min ${NUM_NODES} --max ${NUM_NODES} \
	--afs-name ${FILE_SHARE_NAME} \
	--afs-mount-path extfs \
	--user-name mat \
	--password dnstvxrz \
	--storage-account-name $(STORAGE_ACCOUNT_NAME) \
	--storage-account-key $(azure_storage_key) \
	--config-file ../../cluster_config/cluster.json


submit-keras-intel:
	$(call submit_keras_intel,$(NUM_NODES),$(PROCESSES_PER_NODE),keras-intel-32)

submit-tf-intel:
	$(call submit_tf_intel,$(NUM_NODES),$(PROCESSES_PER_NODE),tf-intel-32)

submit-pytorch:
	$(call submit_pytorch,$(NUM_NODES),$(PROCESSES_PER_NODE),pytorch-32)