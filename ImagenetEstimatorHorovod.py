import logging
import os
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import horovod.tensorflow as hvd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

tf.__version__

WIDTH = 224
HEIGHT = 224
CHANNELS = 3
LR = 0.0001
EPOCHS = 5
BATCHSIZE = 32
IMAGENET_RGB_MEAN_CAFFE = np.array([123.68, 116.78, 103.94], dtype=np.float32)
IMAGENET_SCALE_FACTOR_CAFFE = 0.017
BUFFER = 10


def _preprocess_image_labels(filename, label):
    # load and preprocess the image
    img_decoded = tf.to_float(tf.image.decode_png(tf.read_file(filename), channels=3))
    img_decoded = tf.image.resize_images(img_decoded, [WIDTH, HEIGHT])
    img_centered = tf.subtract(img_decoded, IMAGENET_RGB_MEAN_CAFFE)
    img_rgb = img_centered * IMAGENET_SCALE_FACTOR_CAFFE
    return img_rgb, tf.cast(label, dtype=tf.float32)


def _parse_function_train(filename, label):
    img_rgb, label = _preprocess_image_labels(filename, label)
    # Random crop (from 264x264)
    img_rgb = tf.random_crop(img_rgb, [HEIGHT, WIDTH, CHANNELS])
    # Random flip
    img_rgb = tf.image.random_flip_left_right(img_rgb)
    # Channels-first
    img_rgb = tf.transpose(img_rgb, [2, 0, 1])
    return img_rgb, label


def _parse_function_eval(filename, label):
    img_rgb, label = _preprocess_image_labels(filename, label)
    # Channels-first
    img_rgb = tf.transpose(img_rgb, [2, 0, 1])
    return img_rgb, label


def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.

    # Reference to the tensor named "x" in the input-function.
    x = features

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, WIDTH, HEIGHT, CHANNELS])

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='b1_conv1',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='b1_conv2',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='b2_conv1',
                           filters=128, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='b2_conv2',
                           filters=128, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Third convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='b3_conv1',
                           filters=256, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='b3_conv2',
                           filters=256, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, name='b3_conv3',
                           filters=256, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=2048, activation=tf.nn.relu)

    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=1000)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': y_pred_cls,
            'probabilities': y_pred,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                                       predictions=y_pred_cls,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=metrics,
            loss=loss)

    # Define the optimizer for improving the neural network.
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

    # Get the TensorFlow op for doing a single optimization step.
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)


def main():
    # Horovod: initialize Horovod.
    hvd.init()

    logger.info('Reading training data info')

    data_dir = path.join(os.getenv('AZ_BATCHAI_INPUT_DATASET'), 'imagenet')
    train_df = pd.DataFrame.from_csv(path.join(data_dir, 'train.csv'))
    train_df = train_df.assign(filenames=train_df.filenames.apply(lambda x: path.join(data_dir, 'train', x)))
    # train_df.head()

    logger.info('Reading validation data info')
    validation_df = pd.DataFrame.from_csv(path.join(data_dir, 'validation.csv'))
    validation_df = validation_df.assign(
        filenames=validation_df.filenames.apply(lambda x: path.join(data_dir, 'validation', x)))
    # validation_df.head()
    oh_encoder = OneHotEncoder(sparse=False)
    # train_df[['num_id']].values.shape
    train_labels = oh_encoder.fit_transform(train_df[['num_id']].values).astype(np.uint8)
    validation_labels = oh_encoder.transform(validation_df[['num_id']].values).astype(np.uint8)
    # validation_labels.shape
    train_data = tf.data.Dataset.from_tensor_slices((train_df['filenames'].values, train_labels))
    train_data_transform = tf.contrib.data.map_and_batch(_parse_function_train, BATCHSIZE)
    train_data = (train_data.shuffle(len(train_df))
                  .repeat()
                  .apply(train_data_transform)
                  .prefetch(BUFFER))

    validation_data = tf.data.Dataset.from_tensor_slices((validation_df['filenames'].values, validation_labels))
    validation_data_transform = tf.contrib.data.map_and_batch(_parse_function_eval, BATCHSIZE)
    validation_data = (validation_data.repeat()
                       .apply(validation_data_transform)
                       .prefetch(BUFFER))


    data_length = len(train_df)
    validation_length = len(validation_df)

    def train_input_fn():
        return train_data.make_one_shot_iterator().get_next()

    def validation_input_fn():
        return validation_data.make_one_shot_iterator().get_next()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    rfig = tf.estimator.RunConfig(save_checkpoints_steps=1000, session_config=config)

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = os.getenv('AZ_BATCHAI_OUTPUT_MODEL') if hvd.rank() == 0 else None

    params = {"learning_rate": 1e-4}
    # rfig = tf.estimator.RunConfig(save_checkpoints_steps=1000)
    logger.info('Creating estimator with params: {}'.format(params))
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir=model_dir,
                                   config=rfig)

    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    logger.info('{} {}'.format(hvd.local_rank(), hvd.size()))
    for epoch in range(EPOCHS):
        logger.info('Running epoch {}...'.format(epoch))
        model.train(input_fn=train_input_fn, steps=10, hooks=[bcast_hook])  # data_length//batch_size
        logger.info('Validation...')
        model.evaluate(input_fn=validation_input_fn, steps=10)  # validation_length//batch_size


if __name__ == '__main__':
    main()
