#!/usr/bin/python
# Modified from https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
from datetime import datetime
import os
import random
import sys
import threading
from pathlib import Path

import fire
import numpy as np
import tensorflow as tf

_NUM_THREADS = 2


# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string("labels_file", "", "Labels file")

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

    colorspace = "RGB"
    channels = 3
    image_format = "JPEG"

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": _int64_feature(height),
                "image/width": _int64_feature(width),
                "image/colorspace": _bytes_feature(tf.compat.as_bytes(colorspace)),
                "image/channels": _int64_feature(channels),
                "image/class/label": _int64_feature(label),
                "image/class/text": _bytes_feature(tf.compat.as_bytes(text)),
                "image/format": _bytes_feature(tf.compat.as_bytes(image_format)),
                "image/filename": _bytes_feature(
                    tf.compat.as_bytes(os.path.basename(filename))
                ),
                "image/encoded": _bytes_feature(tf.compat.as_bytes(image_buffer)),
            }
        )
    )
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format="rgb", quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format="rgb", quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(
            self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data}
        )

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data}
        )
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
    # File list from:
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
    # https: // da - data.blogspot.com / 2016 / 02 / cleaning - imagenet - dataset - collected.html
    return "n02105855_2933.JPEG" in filename


def _is_cmyk(filename):
    """Determine if file contains a CMYK JPEG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    # https: // da - data.blogspot.com / 2016 / 02 / cleaning - imagenet - dataset - collected.html
    blacklist = [
        "n01739381_1309.JPEG",
        "n02077923_14822.JPEG",
        "n02447366_23489.JPEG",
        "n02492035_15739.JPEG",
        "n02747177_10752.JPEG",
        "n03018349_4028.JPEG",
        "n03062245_4620.JPEG",
        "n03347037_9675.JPEG",
        "n03467068_12171.JPEG",
        "n03529860_11437.JPEG",
        "n03544143_17228.JPEG",
        "n03633091_5218.JPEG",
        "n03710637_5125.JPEG",
        "n03961711_5286.JPEG",
        "n04033995_2932.JPEG",
        "n04258138_17003.JPEG",
        "n04264628_27969.JPEG",
        "n04336792_7448.JPEG",
        "n04371774_5854.JPEG",
        "n04596742_4225.JPEG",
        "n07583066_647.JPEG",
        "n13037406_4650.JPEG",
    ]
    return filename.split("/")[-1] in blacklist


def _process_image(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.

    Notes:
        See https://da-data.blogspot.com/2016/02/cleaning-imagenet-dataset-collected.html
        for blacklisted items in ImageNet dataset
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, "rb") as f:
        image_data = f.read()

    # Clean the dirty data.
    if _is_png(filename):
        # 1 image is a PNG.
        print("Converting PNG to JPEG for %s" % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        # 22 JPEG images are in CMYK colorspace.
        print("Converting CMYK to RGB for %s" % filename)
        image_data = coder.cmyk_to_rgb(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(
    coder,
    thread_index,
    ranges,
    name,
    filenames,
    texts,
    labels,
    num_shards,
    output_directory,
):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(
        ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1
    ).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print("SKIPPED: Unexpected error while decoding %s." % filename)
                continue

            example = _convert_to_example(
                filename, image_buffer, label, text, height, width
            )
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print(
                    "%s [thread %d]: Processed %d of %d images in thread batch."
                    % (datetime.now(), thread_index, counter, num_files_in_thread)
                )
                sys.stdout.flush()

        writer.close()
        print(
            "%s [thread %d]: Wrote %d images to %s"
            % (datetime.now(), thread_index, shard_counter, output_file)
        )
        sys.stdout.flush()
        shard_counter = 0
    print(
        "%s [thread %d]: Wrote %d images to %d shards."
        % (datetime.now(), thread_index, counter, num_files_in_thread)
    )
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards, output_directory):
    """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)
    num_threads = 2
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (
            coder,
            thread_index,
            ranges,
            name,
            filenames,
            texts,
            labels,
            num_shards,
            output_directory,
        )
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print(
        "%s: Finished writing all %d images in data set."
        % (datetime.now(), len(filenames))
    )
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
    print("Determining list of input files and labels from %s." % data_dir)

    with open(labels_file) as f:
        labels_dict = json.load(f)

    lookup = {
        value[0]: {"noun": value[1], "label": int(key)}
        for key, value in labels_dict.items()
    }

    labels = []
    filenames = []
    texts = []

    # Construct the list of JPEG files and labels.
    path = Path(data_dir)
    for id, values in lookup.items():
        matching_files = glob.glob(str(path / f"{id}" / "*.JPEG"))
        noun = values["noun"]
        label = values["label"]
        labels.extend([label] * len(matching_files))
        texts.extend([noun] * len(matching_files))
        filenames.extend(matching_files)

        print(f"Found {len(matching_files)} for {id}:{label}:{noun}")
        print(f"Total {len(filenames)}")

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(42)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print(
        "Found %d JPEG files across %d labels inside %s."
        % (len(filenames), len(lookup.keys()), data_dir)
    )
    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file, output_directory):
    """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
    os.makedirs(output_directory, exist_ok=True)
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards, output_directory)


def main(
    train_path,
    validation_path,
    output_directory,
    class_index_file,
    train_shards=1014,
    validation_shards=128,
):
    os.makedirs(output_directory, exist_ok=True)
    _process_dataset(
        "validation",
        validation_path,
        validation_shards,
        class_index_file,
        os.path.join(output_directory, "validation"),
    )
    _process_dataset(
        "train",
        train_path,
        train_shards,
        class_index_file,
        os.path.join(output_directory, "train"),
    )


if __name__ == "__main__":
    fire.Fire(main)
