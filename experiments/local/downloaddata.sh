#!/usr/bin/env bash
# Download data
mkdir -p $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/train.tar.gz $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/validation.tar.gz $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/train.csv $AZ_BATCHAI_MOUNT_ROOT/imagenet
rsync --info=progress2 $AZ_BATCHAI_MOUNT_ROOT/nfs/imagenet/validation.csv $AZ_BATCHAI_MOUNT_ROOT/imagenet
cd $AZ_BATCHAI_MOUNT_ROOT/imagenet
tar -xzf train.tar.gz
tar -xzf validation.tar.gz
