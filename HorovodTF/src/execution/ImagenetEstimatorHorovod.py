"""
Trains ResNet50 using Horovod.

It requires the following env variables
AZ_BATCHAI_INPUT_TRAIN
AZ_BATCHAI_INPUT_TEST
AZ_BATCHAI_OUTPUT_MODEL
AZ_BATCHAI_JOB_TEMP_DIR
"""
import logging

logging.basicConfig(level=logging.INFO)

import os
from os import path

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib import slim
from toolz import pipe
from timer import Timer

_WIDTH = 224
_HEIGHT = 224
_CHANNELS = 3
_LR = 0.001
_EPOCHS = 1
_BATCHSIZE = 64
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_BUFFER = 10

def _str_to_bool(in_str):
    if 't' in in_str.lower():
        return True
    else:
        return False

_DISTRIBUTED = _str_to_bool(os.getenv('DISTRIBUTED', 'False'))

if _DISTRIBUTED:
    import horovod.tensorflow as hvd


logger = logging.getLogger(__name__)

resnet_v1_50 = nets.resnet_v1.resnet_v1_50


def _load_image(filename, channels=_CHANNELS):
    return tf.to_float(tf.image.decode_png(tf.read_file(filename), channels=channels))


def _resize(img, width=_WIDTH, height=_HEIGHT):
    return tf.image.resize_images(img, [width, height])


def _centre(img, mean_subtraction=(_R_MEAN, _G_MEAN, _B_MEAN)):
    return tf.subtract(img, list(mean_subtraction))


def _random_crop(img, width=_WIDTH, height=_HEIGHT, channels=_CHANNELS):
    return tf.random_crop(img, [height, width, channels])


def _random_horizontal_flip(img):
    return tf.image.random_flip_left_right(img)


def _preprocess_images(filename):
    return pipe(filename,
                _load_image,
                _resize,
                _centre)


def _preprocess_labels(label):
    return tf.cast(label, dtype=tf.int32)


def _parse_function_train(filename, label):
    img_rgb = pipe(filename,
                   _preprocess_images,
                   _random_crop,
                   _random_horizontal_flip)

    return img_rgb, _preprocess_labels(label)


def _parse_function_eval(filename, label):
    return _preprocess_images(filename), _preprocess_labels(label)


def _get_optimizer(params, is_distributed=_DISTRIBUTED):
    if is_distributed:
        # Horovod: add Horovod Distributed Optimizer.
        return hvd.DistributedOptimizer(tf.train.MomentumOptimizer(learning_rate=params["learning_rate"] * hvd.size(),
                                                                   momentum=0.9))
    else:
        return tf.train.MomentumOptimizer(learning_rate=params["learning_rate"], momentum=0.9)


def model_fn(features, labels, mode, params):
    """
    features: This is the x-arg from the input_fn.
    labels:   This is the y-arg from the input_fn,
              see e.g. train_input_fn for these two.
    mode:     Either TRAIN, EVAL, or PREDICT
    params:   User-defined hyper-parameters, e.g. learning-rate.
    """

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1_50(features,
                                 num_classes=params['classes'])
        logits = tf.reshape(logits, shape=[-1, params['classes']])

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

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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

    optimizer = _get_optimizer(params)

    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)


def _append_path_to(data_path, data_series):
    return data_series.apply(lambda x: path.join(data_path, x))


def _load_training(data_dir):
    train_df = pd.read_csv(path.join(data_dir, 'train.csv'))
    return train_df.assign(filenames=_append_path_to(path.join(data_dir, 'train'),
                                                     train_df.filenames))


def _load_validation(data_dir):
    train_df = pd.read_csv(path.join(data_dir, 'validation.csv'))
    return train_df.assign(filenames=_append_path_to(path.join(data_dir, 'validation'),
                                                     train_df.filenames))


def _create_data_fn(train_path, test_path):
    logger.info('Reading training data info')
    train_df = _load_training(train_path)

    logger.info('Reading validation data info')
    validation_df = _load_validation(test_path)

    train_labels=train_df[['num_id']].values.ravel()-1
    validation_labels=validation_df[['num_id']].values.ravel()-1

    train_data = tf.data.Dataset.from_tensor_slices((train_df['filenames'].values, train_labels))
    train_data_transform = tf.contrib.data.map_and_batch(_parse_function_train, _BATCHSIZE)
    train_data = (train_data.shuffle(len(train_df))
                            .repeat()
                            .apply(train_data_transform)
                            .prefetch(_BUFFER))

    validation_data = tf.data.Dataset.from_tensor_slices((validation_df['filenames'].values, validation_labels))
    validation_data_transform = tf.contrib.data.map_and_batch(_parse_function_eval, _BATCHSIZE)
    validation_data = (validation_data.apply(validation_data_transform)
                                      .prefetch(_BUFFER))

    def _train_input_fn():
        return train_data.make_one_shot_iterator().get_next()

    def _validation_input_fn():
        return validation_data.make_one_shot_iterator().get_next()

    _train_input_fn.length = len(train_df)
    _validation_input_fn.length = len(validation_df)
    _train_input_fn.classes = 1000
    _validation_input_fn.classes = 1000

    return _train_input_fn, _validation_input_fn


def _get_runconfig(is_distributed=_DISTRIBUTED):
    if is_distributed:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        return tf.estimator.RunConfig(save_checkpoints_steps=10000,
                                      session_config=config)
    else:
        return tf.estimator.RunConfig(save_checkpoints_steps=10000)


def _get_model_dir(is_distributed=_DISTRIBUTED):
    if is_distributed:
        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        return os.getenv('AZ_BATCHAI_OUTPUT_MODEL') if hvd.rank() == 0 else os.getenv('AZ_BATCHAI_JOB_TEMP_DIR')
    else:
        return os.getenv('AZ_BATCHAI_OUTPUT_MODEL')


def _get_hooks(is_distributed=_DISTRIBUTED):
    if is_distributed:
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        logger.info('Rank: {} Cluster Size {}'.format(hvd.local_rank(), hvd.size()))
        return [bcast_hook]
    else:
        return []


def _is_master(is_distributed=_DISTRIBUTED):
    if is_distributed:
        if hvd.rank() == 0:
            return True
        else:
            return False
    else:
        return True


def main():
    if _DISTRIBUTED:
        # Horovod: initialize Horovod.
        logger.info("Runnin Distributed")
        hvd.init()
    logger.info("Tensorflow version {}".format(tf.__version__))
    train_input_fn, validation_input_fn = _create_data_fn(os.getenv('AZ_BATCHAI_INPUT_TRAIN'),
                                                          os.getenv('AZ_BATCHAI_INPUT_TEST'))

    run_config = _get_runconfig()
    model_dir = _get_model_dir()

    params = {"learning_rate": _LR,
              "classes": train_input_fn.classes}
    logger.info('Creating estimator with params: {}'.format(params))
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir=model_dir,
                                   config=run_config)

    hooks = _get_hooks()

    with Timer(output=logger.info, prefix="Training"):
        logger.info('Training...')
        model.train(input_fn=train_input_fn,
                    steps=_EPOCHS * train_input_fn.length // (_BATCHSIZE*hvd.size()),
                    hooks=hooks)

    if _is_master():
        with Timer(output=logger.info, prefix="Testing"):
            logger.info('Testing...')
            model.evaluate(input_fn=validation_input_fn)


if __name__ == '__main__':
    main()