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

define submit_keras_local
	$(call generate_job_local,masalvar/horovod-keras:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py,$(1), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(2))
endef

define submit_tf_intel
	$(call generate_job_intel,masalvar/horovod-intel:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_tf
	$(call generate_job_openmpi,masalvar/horovod:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_tf_local
	$(call generate_job_local,masalvar/horovod-intel:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(2))
endef

define submit_pytorch
	$(call generate_job_openmpi,masalvar/horovod-pytorch,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_horovod.py,$(1),$(2), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(3))
endef

define submit_pytorch_local
	$(call generate_job_local,masalvar/horovod-pytorch,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_horovod.py,$(1), --synthetic_length ${FAKE_DATA_LENGTH})
	$(call submit_job, $(2))
endef


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


submit-all: submit-keras-intel32 submit-keras-intel16 submit-keras-intel8 submit-keras-intel4 submit-tf-intel32 \
submit-tf-intel16 submit-tf-intel8 submit-tf-intel4 submit-pytorch32 submit-pytorch16 submit-pytorch8 submit-pytorch4 \
submit-keras-local submit-tf-local submit-pytorch-local

submit-keras-intel32:
	$(call submit_keras_intel,$(NUM_NODES),$(PROCESSES_PER_NODE),keras-intel-32)

submit-keras-intel16:
	$(call submit_keras_intel,4,$(PROCESSES_PER_NODE),keras-intel-16)

submit-keras-intel8:
	$(call submit_keras_intel,2,$(PROCESSES_PER_NODE),keras-intel-8)

submit-keras-intel4:
	$(call submit_keras_intel,1,$(PROCESSES_PER_NODE),keras-intel-4)

submit-keras-local:
	$(call submit_keras_local,1,keras-local)


submit-tf-intel32:
	$(call submit_tf_intel,$(NUM_NODES),$(PROCESSES_PER_NODE),tf-intel-32)

submit-tf-intel16:
	$(call submit_tf_intel,4,$(PROCESSES_PER_NODE),tf-intel-16)

submit-tf-intel8:
	$(call submit_tf_intel,2,$(PROCESSES_PER_NODE),tf-intel-8)

submit-tf-intel4:
	$(call submit_tf_intel,1,$(PROCESSES_PER_NODE),tf-intel-4)

submit-tf-local:
	$(call submit_tf_local,1,tf-local)


submit-pytorch32:
	$(call submit_pytorch,8,$(PROCESSES_PER_NODE),pytorch-32)

submit-pytorch16:
	$(call submit_pytorch,4,$(PROCESSES_PER_NODE),pytorch-16)

submit-pytorch8:
	$(call submit_pytorch,2,$(PROCESSES_PER_NODE),pytorch-8)

submit-pytorch4:
	$(call submit_pytorch,1,$(PROCESSES_PER_NODE),pytorch-4)

submit-pytorch-local:
	$(call submit_pytorch_local,1,pytorch-local)
