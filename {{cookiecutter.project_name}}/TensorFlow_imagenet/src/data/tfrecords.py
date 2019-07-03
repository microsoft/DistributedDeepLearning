import logging
import os

import horovod.tensorflow as hvd
import tensorflow as tf

import defaults
import imagenet_preprocessing


def get_filenames(is_training, data_dir, num_files=1014):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, "train-%05d-of-01014" % i) for i in range(num_files)
        ]
    else:
        return [
            os.path.join(data_dir, "validation-%05d-of-00128" % i) for i in range(128)
        ]


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
          Example protocol buffer.

    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
    """
    feature_map = {
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/class/label": tf.io.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1
        ),
        "image/class/text": tf.io.FixedLenFeature(
            [], dtype=tf.string, default_value=""
        ),
    }

    features = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map
    )
    label = tf.cast(features["image/class/label"], dtype=tf.int32)
    return features["image/encoded"], label


def parse_record(
    raw_record,
    is_training,
    dtype,
    data_format="channels_last",
    image_size=defaults.DEFAULT_IMAGE_SIZE,
    num_channels=defaults.NUM_CHANNELS,
):
    """Parses a record containing a training example of an image.
    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
        raw_record: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
        is_training: A boolean denoting whether the input is for training.
        dtype: data type to use for images/features.
        data_format: the axis order of the matrix, channels_last NHWC or channels_first NCHW

    Returns:
        Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, label = _parse_example_proto(raw_record)

    image = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        output_height=image_size,
        output_width=image_size,
        num_channels=num_channels,
        is_training=is_training,
        data_format=data_format,
    )
    image = tf.cast(image, dtype)

    return image, label


def input_fn(
    is_training,
    data_dir,
    batch_size,
    repetitions=1,
    dtype=tf.float32,
    num_parallel_batches=1,
    parse_record_fn=parse_record,
    data_format="channels_last",
    distributed=False,
    file_shuffle_buffer=5,
    data_shuffle_buffer=defaults.SHUFFLE_BUFFER,
):
    """Input function which provides batches for train or eval.

    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: The directory containing the input data.
        batch_size: The number of samples per batch.
        repetitions: The number times to repeat the dataset.
        dtype: Data type to use for images/features
        num_parallel_batches: Number of parallel batches for tf.data.
        parse_record_fn: Function to use for parsing the records.

    Returns:
        A dataset that can be used for iteration.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading data info from {data_dir}")
    filenames = get_filenames(is_training, data_dir)
    for f in filenames:
        if not os.path.exists(f):
            raise ValueError(f"{f} File doesn't exist ")
    logger.info(f"Found {len(filenames)} files")
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        if distributed:
            dataset = dataset.shard(hvd.size(), hvd.rank())

        dataset = dataset.shuffle(buffer_size=file_shuffle_buffer)  # _NUM_TRAIN_FILES

    # Convert to individual records.
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=num_parallel_batches,
            buffer_output_elements=5,
        )
    )

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=data_shuffle_buffer,
        parse_record_fn=parse_record_fn,
        repetitions=repetitions,
        dtype=dtype,
        num_parallel_batches=num_parallel_batches,
        data_format=data_format,
    )


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           repetitions=1,
                           dtype=tf.float32,
                           data_format="channels_last",
                           num_parallel_batches=1):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    repetitions: The number of times to repeat the dataset.
    dtype: Data type to use for images/features.
    num_parallel_batches: Number of parallel batches for tf.data.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.
  dataset = dataset.repeat(repetitions)

  # Parses the raw records into images and labels.
  dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dtype, data_format=data_format),
          batch_size=batch_size,
          num_parallel_batches=num_parallel_batches,
          drop_remainder=False))


  dataset = dataset.prefetch(buffer_size=256)

  return dataset