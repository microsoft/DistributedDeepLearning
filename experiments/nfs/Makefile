include ../experiments_config.mk

NFS_NAME=batch${ID}nfs
EXPERIMENT:=experiment_imagenet_${GPU_TYPE}
NFS_IP:=""

include ../../include/control.mk

define submit_keras_intel
	$(call generate_job_intel,masalvar/horovod-intel-keras:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py,$(1),$(2),$(3))
	$(call submit_job,$(4))
endef

define submit_keras
	$(call generate_job_openmpi,masalvar/horovod-keras:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_keras_horovod.py,$(1),$(2),$(3))
	$(call submit_job,$(4))
endef

define submit_tf_intel
	$(call generate_job_intel,masalvar/horovod-intel:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2),$(3))
	$(call submit_job,$(4))
endef

define submit_tf
	$(call generate_job_openmpi,masalvar/horovod:9-1.8-.13.2,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_estimator_tf_horovod.py,$(1),$(2),$(3))
	$(call submit_job,$(4))
endef

define submit_pytorch
	$(call generate_job_openmpi,masalvar/horovod-pytorch,\$$AZ_BATCHAI_INPUT_SCRIPTS/imagenet_pytorch_horovod.py,$(1),$(2),$(3))
	$(call submit_job,$(4))
endef

define upload-data-nfs
	scp nodeprep.sh mat@$(1):~/
	ssh mat@$(1) sudo chmod 777 ~/nodeprep.sh
	ssh mat@$(1) ./nodeprep.sh
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
	--nfs ${NFS_NAME} \
	--nfs-mount-path nfs \
	--config-file ../../cluster_config/cluster.json

upload-data-nfs:
	$(call upload-data-nfs,$(NFS_IP))

submit-keras-intel:
	$(call submit_keras_intel,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet ,keras-intel-32)

submit-tf-intel:
	$(call submit_tf_intel,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet ,tf-intel-32)

submit-pytorch:
	$(call submit_pytorch,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet ,pytorch-32)