include ../experiments_config.mk

CONTAINER_NAME=batch${ID}blob
EXPERIMENT:=experiment_imagenet_blob_${GPU_TYPE}

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


upload-data: upload-training upload-validation upload-csv
	@echo 'All data uploaded'

upload-training: set-storage
	azcopy --source https://datasharesa.blob.core.windows.net/imagenet/train \
			--source-key 'owUPSqTbwAigV54BHTr8oYABEha8xi/VsA4HD06GboDgOb3pf6OFgtw/tlKYv/AlkgSIBkxqoA28hnkIeo4NFg==' \
			--destination  https://${azure_storage_account}.blob.core.windows.net/${CONTAINER_NAME}/train \
			--dest-key ${azure_storage_key} --quiet --recursive --exclude-older

upload-validation: set-storage
	azcopy --source https://datasharesa.blob.core.windows.net/imagenet/validation \
		--source-key 'owUPSqTbwAigV54BHTr8oYABEha8xi/VsA4HD06GboDgOb3pf6OFgtw/tlKYv/AlkgSIBkxqoA28hnkIeo4NFg==' \
		--destination  https://${azure_storage_account}.blob.core.windows.net/${CONTAINER_NAME}/validation \
		--dest-key ${azure_storage_key} --quiet --recursive


upload-csv: set-storage
	azcopy --source https://datasharesa.blob.core.windows.net/imagenet/train.csv \
			--source-key 'owUPSqTbwAigV54BHTr8oYABEha8xi/VsA4HD06GboDgOb3pf6OFgtw/tlKYv/AlkgSIBkxqoA28hnkIeo4NFg==' \
			--destination  https://${azure_storage_account}.blob.core.windows.net/${CONTAINER_NAME}/train.csv \
			--dest-key ${azure_storage_key} --quiet

	azcopy --source https://datasharesa.blob.core.windows.net/imagenet/validation.csv \
			--source-key 'owUPSqTbwAigV54BHTr8oYABEha8xi/VsA4HD06GboDgOb3pf6OFgtw/tlKYv/AlkgSIBkxqoA28hnkIeo4NFg==' \
			--destination  https://${azure_storage_account}.blob.core.windows.net/${CONTAINER_NAME}/validation.csv \
			--dest-key ${azure_storage_key} --quiet

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
	---container-name $(CONTAINER_NAME) \
	--container-mount-path extcn \
	--config-file ../../cluster_config/cluster.json


submit-keras-intel:
	$(call submit_keras_intel,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/extcn/imagenet ,keras-intel-32)

submit-tf-intel:
	$(call submit_tf_intel,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/extcn/imagenet ,tf-intel-32)

submit-pytorch:
	$(call submit_pytorch,$(NUM_NODES),$(PROCESSES_PER_NODE), --data \$$AZ_BATCHAI_MOUNT_ROOT/extcn/imagenet ,pytorch-32)