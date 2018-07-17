# Variables for Batch AI - change as necessary
ID:=disdl
LOCATION:=eastus
GROUP_NAME:=batch${ID}rg
STORAGE_ACCOUNT_NAME:=batch${ID}st
FILE_SHARE_NAME:=batch${ID}share
SELECTED_SUBSCRIPTION:="Team Danielle Internal"
WORKSPACE:=workspace

VM_SIZE:=Standard_NC24rs_v3
NUM_NODES:=8
CLUSTER_NAME:=msv100


GPU_TYPE:=V100
PROCESSES_PER_NODE:=4
