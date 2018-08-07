define PROJECT_HELP_MSG
Usage:
    make help                   show this message
    make build                  build docker image
    make push					 push container
    make run					 run benchmarking container
    make setup                  setup the cluster
    make show-cluster
    make list-clusters
    make run-bait-intel         run batch ai benchamrk using intel mpi
    make run-bait-openmpi       run batch ai benchmark using open mpi
    make run-bait-local         run batch ai benchmark on one node
    make list-jobs
    make list-files
    make stream-stdout
    make stream-stderr
    make delete-job
    make delete-cluster
    make delete                 delete everything including experiments, workspace and resource group
endef
export PROJECT_HELP_MSG


define generate_job_intel
 python ../generate_job_spec.py $(1) intelmpi \
 	$(2) \
 	--filename job.json \
 	--node_count $(3) \
 	--ppn $(4) \
 	$(5)
endef


define generate_job_openmpi
 python ../generate_job_spec.py $(1) openmpi \
 	$(2) \
 	--filename job.json \
 	--node_count $(3) \
 	--ppn $(4) \
 	$(5)
endef


define generate_job_local
 python ../generate_job_spec.py $(1) local \
 	$(2) \
 	--filename job.json \
 	--node_count 1 \
 	--ppn $(3) \
 	$(4)
endef


define stream_stdout
	az batchai job file stream -w $(WORKSPACE) -e $(EXPERIMENT) \
	--j $(1) --output-directory-id stdouterr -f stdout.txt
endef


define submit_job
	az batchai job create -n $(1) --cluster ${CLUSTER_NAME} -w $(WORKSPACE) -e $(EXPERIMENT) -f job.json
endef

define delete_job
	az batchai job delete -w $(WORKSPACE) -e $(EXPERIMENT) --name $(1) -y
endef

define upload_script
	az storage file upload --share-name ${FILE_SHARE_NAME} --source $(1) --path scripts --account-name $(azure_storage_account) --account-key $(azure_storage_key)
endef

select-subscription:
	az login -o table
	az account set --subscription $(SELECTED_SUBSCRIPTION)

create-resource-group:
	az group create -n $(GROUP_NAME) -l $(LOCATION) -o table

create-storage:
	@echo "Creating storage account"
	az storage account create -l $(LOCATION) -n $(STORAGE_ACCOUNT_NAME) -g $(GROUP_NAME) --sku Standard_LRS

set-storage:
	$(eval azure_storage_key:=$(shell az storage account keys list -n $(STORAGE_ACCOUNT_NAME) -g $(GROUP_NAME) | jq '.[0]["value"]'))
	$(eval azure_storage_account:= $(STORAGE_ACCOUNT_NAME))
	$(eval file_share_name:= $(FILE_SHARE_NAME))

set-az-defaults:
	az configure --defaults location=${LOCATION}
	az configure --defaults group=${GROUP_NAME}

create-fileshare: set-storage
	@echo "Creating fileshare"
	az storage share create -n $(file_share_name) --account-name $(azure_storage_account) --account-key $(azure_storage_key)

create-directory: set-storage
	az storage directory create --share-name $(file_share_name)  --name scripts --account-name $(azure_storage_account) --account-key $(azure_storage_key)

create-nfs:
	az batchai file-server create -n $(NFS_NAME) -w ${WORKSPACE} --disk-count 4 --disk-size 250 -s Standard_DS4_v2 -u mat -p d13NHAL! -g ${GROUP_NAME} --storage-sku Premium_LRS

list-nfs:
	az batchai file-server list -o table -w ${WORKSPACE} -g ${GROUP_NAME}

create-container: set-storage
	az storage container create --account-name $(azure_storage_account) --account-key $(azure_storage_key) --name ${CONTAINER_NAME}

upload-scripts: set-storage
	$(call upload_script, ../../HorovodKeras/src/data_generator.py)
	$(call upload_script, ../../HorovodKeras/src/imagenet_keras_horovod.py)
	$(call upload_script, ../../HorovodTF/src/imagenet_estimator_tf_horovod.py)
	$(call upload_script, ../../HorovodTF/src/resnet_model.py)
	$(call upload_script, ../../HorovodPytorch/src/imagenet_pytorch_horovod.py)
	$(call upload_script, ../../common/timer.py)

upload-nodeprep-scripts: set-storage
	$(call upload_script, ../../cluster_config/docker.service)
	$(call upload_script, ../../cluster_config/nodeprep.sh)

create-workspace:
	az batchai workspace create -n $(WORKSPACE) -g $(GROUP_NAME)

create-experiment:
	az batchai experiment create -n $(EXPERIMENT) -g $(GROUP_NAME) -w $(WORKSPACE)

show-cluster:
	az batchai cluster show -n ${CLUSTER_NAME} -w $(WORKSPACE)

list-clusters:
	az batchai cluster list -w $(WORKSPACE) -o table

list-nodes:
	az batchai cluster node list -c ${CLUSTER_NAME} -w $(WORKSPACE)

list-jobs:
	az batchai job list -w $(WORKSPACE) -e $(EXPERIMENT) -o table

list-files:
	az batchai job file list -w $(WORKSPACE) -e $(EXPERIMENT) --j ${JOB_NAME} --output-directory-id stdouterr

stream-stdout:
	$(call stream_stdout, ${JOB_NAME})

stream-stderr:
	az batchai job file stream -w $(WORKSPACE) -e $(EXPERIMENT) --j ${JOB_NAME} --output-directory-id stdouterr -f stderr.txt

delete-job:
	$(call delete_job, ${JOB_NAME})

delete-cluster:
	az configure --defaults group=''
	az configure --defaults location=''
	az batchai cluster delete -w $(WORKSPACE) --name ${CLUSTER_NAME} -g ${GROUP_NAME} -y

delete: delete-cluster
	az batchai experiment delete -w $(WORKSPACE) --name ${experiment} -g ${GROUP_NAME} -y
	az batchai workspace delete -w ${WORKSPACE} -g ${GROUP_NAME} -y
	az group delete --name ${GROUP_NAME} -y


setup: select-subscription create-resource-group create-workspace create-storage set-storage set-az-defaults create-fileshare create-cluster list-clusters
	@echo "Cluster created"

#
####### Submit Jobs ######
#
submit-all: submit-keras-intel32 submit-keras-intel16 submit-keras-intel8 submit-keras-intel4 submit-tf-intel32 \
submit-tf-intel16 submit-tf-intel8 submit-tf-intel4 submit-pytorch32 submit-pytorch16 submit-pytorch8 submit-pytorch4 \
submit-keras-local submit-tf-local submit-pytorch-local

clean-jobs:
	$(call delete_job, tf-local)
	$(call delete_job, tf-intel-4)
	$(call delete_job, tf-intel-8)
	$(call delete_job, tf-intel-16)
	$(call delete_job, tf-intel-32)
	
	$(call delete_job, keras-local)
	$(call delete_job, keras-intel-4)
	$(call delete_job, keras-intel-8)
	$(call delete_job, keras-intel-16)
	$(call delete_job, keras-intel-32)
	
	$(call delete_job, pytorch-local)
	$(call delete_job, pytorch-4)
	$(call delete_job, pytorch-8)
	$(call delete_job, pytorch-16)
	$(call delete_job, pytorch-32)

####### Gather Results ######

gather-results:results.json
	@echo "All results gathered"

results.json: pytorch_1gpulocal_$(GPU_TYPE)_local.results pytorch_4gpuopen_$(GPU_TYPE)_open.results \
			  pytorch_8gpuopen_$(GPU_TYPE)_open.results pytorch_16gpuopen_$(GPU_TYPE)_open.results \
			  pytorch_32gpuopen_$(GPU_TYPE)_open.results \
			  tf_1gpulocal_$(GPU_TYPE)_local.results tf_4gpuintel_$(GPU_TYPE)_intel.results \
			  tf_8gpuintel_$(GPU_TYPE)_intel.results tf_16gpuintel_$(GPU_TYPE)_intel.results \
			  tf_32gpuintel_$(GPU_TYPE)_intel.results \
			  keras_1gpulocal_$(GPU_TYPE)_local.results keras_4gpuintel_$(GPU_TYPE)_intel.results \
			  keras_8gpuintel_$(GPU_TYPE)_intel.results keras_16gpuintel_$(GPU_TYPE)_intel.results \
			  keras_32gpuintel_$(GPU_TYPE)_intel.results 
	python ../parse_results.py
	

pytorch_1gpulocal_$(GPU_TYPE)_local.results:
	$(call stream_stdout, pytorch-local)>pytorch_1gpulocal_$(GPU_TYPE)_local.results

pytorch_4gpuopen_$(GPU_TYPE)_open.results:
	$(call stream_stdout, pytorch-4)>pytorch_4gpuopen_$(GPU_TYPE)_open.results

pytorch_8gpuopen_$(GPU_TYPE)_open.results:
	$(call stream_stdout, pytorch-8)>pytorch_8gpuopen_$(GPU_TYPE)_open.results

pytorch_16gpuopen_$(GPU_TYPE)_open.results:
	$(call stream_stdout, pytorch-16)>pytorch_16gpuopen_$(GPU_TYPE)_open.results

pytorch_32gpuopen_$(GPU_TYPE)_open.results:
	$(call stream_stdout, pytorch-32)>pytorch_32gpuopen_$(GPU_TYPE)_open.results




tf_1gpulocal_$(GPU_TYPE)_local.results:
	$(call stream_stdout, tf-local)>tf_1gpulocal_$(GPU_TYPE)_local.results

tf_4gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, tf-intel-4)>tf_4gpuintel_$(GPU_TYPE)_intel.results

tf_8gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, tf-intel-8)>tf_8gpuintel_$(GPU_TYPE)_intel.results

tf_16gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, tf-intel-16)>tf_16gpuintel_$(GPU_TYPE)_intel.results

tf_32gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, tf-intel-32)>tf_32gpuintel_$(GPU_TYPE)_intel.results



keras_1gpulocal_$(GPU_TYPE)_local.results:
	$(call stream_stdout, keras-local)>keras_1gpulocal_$(GPU_TYPE)_local.results

keras_4gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, keras-intel-4)>keras_4gpuintel_$(GPU_TYPE)_intel.results

keras_8gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, keras-intel-8)>keras_8gpuintel_$(GPU_TYPE)_intel.results

keras_16gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, keras-intel-16)>keras_16gpuintel_$(GPU_TYPE)_intel.results

keras_32gpuintel_$(GPU_TYPE)_intel.results:
	$(call stream_stdout, keras-intel-32)>keras_32gpuintel_$(GPU_TYPE)_intel.results
	

clean-results:
	rm results.json
	rm *.results

make plot: results.json
	python ../produce_plot.py

.PHONY: help build push
