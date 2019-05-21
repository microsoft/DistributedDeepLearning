"""Prepare the ImageNet dataset"""

import hashlib
import os
import tarfile

import fire
import pandas as pd
from tqdm import tqdm
import logging

_TRAIN_TAR = "ILSVRC2012_img_train.tar"
_TRAIN_TAR_SHA1 = "43eda4fe35c1705d6606a6a7a633bc965d194284"
_VAL_TAR = "ILSVRC2012_img_val.tar"
_VAL_TAR_SHA1 = "5f3f73da3395154b60528b2b2a2caf2374f5f178"


def _sha1(filename, blocksize=65536):
    hash_sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()


def check_sha1(filename, sha1_hash):
    hash_digits = _sha1(filename)
    return hash_digits == sha1_hash


def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError("File not found: " + filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError("Corrupted file: " + filename)


def _extract_train(tar_fname, target_dir):
    logger = logging.getLogger(__name__)
    os.makedirs(target_dir)
    with tarfile.open(tar_fname) as tar:
        logger.info("Extracting " + tar_fname + "...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description("Extract " + class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()


def _extract_val(tar_fname, target_dir):
    logger = logging.getLogger(__name__)
    os.makedirs(target_dir)
    logger.info("Extracting " + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)

    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), "imagenet_val_maps.csv")
    df = pd.read_csv(val_maps_file)
    for d in df['class'].unique():
        os.makedirs(os.path.join(target_dir, d))
    for index, row in df.iterrows():
        os.rename(os.path.join(target_dir, row['filename']), os.path.join(target_dir, row['class'], row['filename']))


def main(download_dir, target_dir, checksum=False):
    target_dir = os.path.expanduser(target_dir)
    download_dir = os.path.expanduser(download_dir)
    train_tar_fname = os.path.join(download_dir, _TRAIN_TAR)
    check_file(train_tar_fname, checksum, _TRAIN_TAR_SHA1)
    val_tar_fname = os.path.join(download_dir, _VAL_TAR)
    check_file(val_tar_fname, checksum, _VAL_TAR_SHA1)

    _extract_train(train_tar_fname, os.path.join(target_dir, "train"))
    _extract_val(val_tar_fname, os.path.join(target_dir, "validation"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
