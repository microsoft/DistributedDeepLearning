import tensorflow as tf


def get_synth_input_fn(height, width, num_channels, num_classes, dtype=tf.float32):
    """Returns an input function that returns a dataset with random data.
    This input_fn returns a data set that iterates over a set of random data and
    bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
    copy is still included. This used to find the upper throughput bound when
    tunning the full input pipeline.

    Args:
        height: Integer height that will be used to create a fake image tensor.
        width: Integer width that will be used to create a fake image tensor.
        num_channels: Integer depth that will be used to create a fake image tensor.
        num_classes: Number of classes that should be represented in the fake labels
          tensor
        dtype: Data type for features/images.

    Returns:
        An input_fn that can be used in place of a real one to return a dataset
        that can be used for iteration.
    """

    def input_fn(
        is_training, data_dir, batch_size, *args, data_format="channels_last", **kwargs
    ):
        """Returns dataset filled with random data."""
        # Synthetic input should be within [0, 255].
        if data_format == "channels_last":
            shape = [height, width, num_channels]
        else:
            shape = [num_channels, height, width]
        inputs = tf.random.truncated_normal(
            [batch_size] + shape,
            dtype=dtype,
            mean=127,
            stddev=60,
            name="synthetic_inputs",
        )

        labels = tf.random.uniform(
            [batch_size],
            minval=0,
            maxval=num_classes - 1,
            dtype=tf.int32,
            name="synthetic_labels",
        )
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
        data = data.prefetch(buffer_size=1024)
        return data

    return input_fn