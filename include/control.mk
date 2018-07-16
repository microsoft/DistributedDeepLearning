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
 python ../../generate_job_spec.py $(1) intelmpi \
 	$(2) \
 	--filename job.json \
 	--node_count $(3) \
 	--ppn $(4) \
 	$(5)
endef


define generate_job_openmpi
 python ../../generate_job_spec.py $(1) openmpi \
 	$(2) \
 	--filename job.json \
 	--node_count $(3) \
 	--ppn $(4) \
 	$(5)
endef


define generate_job_local
 python ../generate_job_spec.py $(1) local \
 	--filename job.json \
 	--node_count 1 \
 	--model $(2) \
 	--ppn $(3)
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

#


#
#!az batchai cluster create \
#--name nc24r \
#--image UbuntuLTS \
#--vm-size Standard_NC24r \
#--min 4 --max 4 \
#--afs-name $FILESHARE_NAME \
#--afs-mount-path extfs \
#--user-name mat \
#--password dnstvxrz \
#--storage-account-name $STORAGE_ACCOUNT_NAME \
#--storage-account-key $storage_account_key \
#--nfs $NFS_NAME \
#--nfs-mount-path nfs

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
	az batchai cluster list-nodes -n ${CLUSTER_NAME} -w $(WORKSPACE) -o table

#run-bait-intel:
#	$(call generate_job_intel,$(intel-image),$(script),$(NUM_NODES),$(PROCESSES_PER_NODE))
#	$(call submit_job, ${JOB_NAME})
#
#run-bait-openmpi:
#	$(call generate_job_openmpi,$(open-image),$(script),$(NUM_NODES),$(PROCESSES_PER_NODE))
#	$(call submit_job, ${JOB_NAME})
#
#run-bait-local:
#	$(call generate_job_local, $(MODEL), $(PROCESSES_PER_NODE))
#	$(call submit_job, ${JOB_NAME})

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
#submit-jobs:
#
#	# Intel Jobs
#	# 1gpuintel
#	$(call generate_job_intel, 1, $(MODEL), 1)
#	$(call submit_job, 1gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 1, $(MODEL), 2)
#	$(call submit_job, 2gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 1, $(MODEL), 3)
#	$(call submit_job, 3gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 1, $(MODEL), 4)
#	$(call submit_job, 4gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 2, $(MODEL), 4)
#	$(call submit_job, 8gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 4, $(MODEL), 4)
#	$(call submit_job, 16gpuintel)
#
#	# 1gpuintel
#	$(call generate_job_intel, 8, $(MODEL), 4)
#	$(call submit_job, 32gpuintel)
#
#	# OpenMPI Jobs
#	# 1gpuopen
#	$(call generate_job_openmpi, 1, $(MODEL), 1)
#	$(call submit_job, 1gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 1, $(MODEL), 2)
#	$(call submit_job, 2gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 1, $(MODEL), 3)
#	$(call submit_job, 3gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 1, $(MODEL), 4)
#	$(call submit_job, 4gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 2, $(MODEL), 4)
#	$(call submit_job, 8gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 4, $(MODEL), 4)
#	$(call submit_job, 16gpuopen)
#
#	# 1gpuopen
#	$(call generate_job_openmpi, 8, $(MODEL), 4)
#	$(call submit_job, 32gpuopen)
#
#	# Local
#	# 1gpulocal
#	$(call generate_job_local, $(MODEL), 1)
#	$(call submit_job, 1gpulocal)
#
#clean-jobs:
#	$(call delete_job, 1gpuintel)
#	$(call delete_job, 2gpuintel)
#	$(call delete_job, 3gpuintel)
#	$(call delete_job, 4gpuintel)
#	$(call delete_job, 8gpuintel)
#	$(call delete_job, 16gpuintel)
#	$(call delete_job, 32gpuintel)
#
#	$(call delete_job, 1gpuopen)
#	$(call delete_job, 2gpuopen)
#	$(call delete_job, 3gpuopen)
#	$(call delete_job, 4gpuopen)
#	$(call delete_job, 8gpuopen)
#	$(call delete_job, 16gpuopen)
#	$(call delete_job, 32gpuopen)
#
#	$(call delete_job, 1gpulocal)
#
#
####### Gather Results ######
#
#gather-results:results.json
#	@echo "All results gathered"
#
#results.json: 1gpulocal_$(GPU_TYPE)_local.results 1gpuintel_$(GPU_TYPE)_intel.results 2gpuintel_$(GPU_TYPE)_intel.results 3gpuintel_$(GPU_TYPE)_intel.results \
#4gpuintel_$(GPU_TYPE)_intel.results 8gpuintel_$(GPU_TYPE)_intel.results 16gpuintel_$(GPU_TYPE)_intel.results 32gpuintel_$(GPU_TYPE)_intel.results \
#1gpuopen_$(GPU_TYPE)_open.results 2gpuopen_$(GPU_TYPE)_open.results 3gpuopen_$(GPU_TYPE)_open.results 4gpuopen_$(GPU_TYPE)_open.results 8gpuopen_$(GPU_TYPE)_open.results \
#16gpuopen_$(GPU_TYPE)_open.results 32gpuopen_$(GPU_TYPE)_open.results
#	python ../parse_results.py
#
#
#1gpulocal_$(GPU_TYPE)_local.results:
#	$(call stream_stdout, 1gpulocal)>1gpulocal_$(GPU_TYPE)_local.results
#
#
#
#1gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 1gpuintel)>1gpuintel_$(GPU_TYPE)_intel.results
#
#2gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 2gpuintel)>2gpuintel_$(GPU_TYPE)_intel.results
#
#3gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 3gpuintel)>3gpuintel_$(GPU_TYPE)_intel.results
#
#4gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 4gpuintel)>4gpuintel_$(GPU_TYPE)_intel.results
#
#8gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 8gpuintel)>8gpuintel_$(GPU_TYPE)_intel.results
#
#16gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 16gpuintel)>16gpuintel_$(GPU_TYPE)_intel.results
#
#32gpuintel_$(GPU_TYPE)_intel.results:
#	$(call stream_stdout, 32gpuintel)>32gpuintel_$(GPU_TYPE)_intel.results
#
#
#
#1gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 1gpuopen)>1gpuopen_$(GPU_TYPE)_open.results
#
#2gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 2gpuopen)>2gpuopen_$(GPU_TYPE)_open.results
#
#3gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 3gpuopen)>3gpuopen_$(GPU_TYPE)_open.results
#
#4gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 4gpuopen)>4gpuopen_$(GPU_TYPE)_open.results
#
#8gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 8gpuopen)>8gpuopen_$(GPU_TYPE)_open.results
#
#16gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 16gpuopen)>16gpuopen_$(GPU_TYPE)_open.results
#
#32gpuopen_$(GPU_TYPE)_open.results:
#	$(call stream_stdout, 32gpuopen)>32gpuopen_$(GPU_TYPE)_open.results
#
#clean-results:
#	rm results.json
#	rm *.results
#
#make plot: results.json
#	python ../produce_plot.py

.PHONY: help build push
