# Variables for Batch AI - change as necessary
ID:=ddl
LOCATION:=eastus
GROUP_NAME:=batch${ID}rg
STORAGE_ACCOUNT_NAME:=batch${ID}st

FILE_SHARE_NAME:=batch${ID}share
VM_SIZE:=Standard_NC24rs_v3
NUM_NODES:=8
CLUSTER_NAME:=msv100

SELECTED_SUBSCRIPTION:="Team Danielle Internal"
WORKSPACE:=workspace
GPU_TYPE:=V100
EXPERIMENT:=experiment_${GPU_TYPE}
PROCESSES_PER_NODE:=4

FAKE_DATA_LENGTH:=1281167

JOB_NAME:=horovod_keras
FAKE:='False'
CONTAINER_NAME:=batch${ID}container