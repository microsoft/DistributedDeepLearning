#!/usr/bin/env bash

# Docker config
sudo cp $AZ_BATCHAI_MOUNT_ROOT/extfs/scripts/docker.service /lib/systemd/system
sudo systemctl daemon-reload
sudo systemctl restart docker

df -h

ls $AZ_BATCHAI_MOUNT_ROOT/nfs/

# Download data
mkdir -p $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/train.tar.gz $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/validation.tar.gz $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/train.csv $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/validation.csv $AZ_BATCHAI_MOUNT_ROOT/imagenet
cd $AZ_BATCHAI_MOUNT_ROOT/imagenet
tar -xzf train.tar.gz
tar -xzf validation.tar.gz
