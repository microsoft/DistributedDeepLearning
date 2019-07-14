""" This the script is the main entry point for training ResNet model using TensorFlow with Horovod

"""
import logging
import logging.config
import os

import fire
import tensorflow as tf

from data.synthetic import get_synth_input_fn
from data import tfrecords, images
from resnet_model import resnet_v1
from timer import Timer
from utils import ExamplesPerSecondHook
import defaults


if defaults.DISTRIBUTED:
    import horovod.tensorflow as hvd


def _get_rank():
    if defaults.DISTRIBUTED:
        try:
            return hvd.rank()
        except:
            return 0
    else:
        return 0


# Data processing
###############################################################################


def _get_optimizer(params, is_distributed=defaults.DISTRIBUTED):
    if is_distributed:
        # Horovod: add Horovod Distributed Optimizer.
        return hvd.DistributedOptimizer(
            tf.train.MomentumOptimizer(
                learning_rate=params["learning_rate"] * hvd.size(),
                momentum=params["momentum"],
            )
        )
    else:
        return tf.train.MomentumOptimizer(
            learning_rate=params["learning_rate"], momentum=params["momentum"]
        )


def build_network(features, mode, params):
    """ Build ResNet50 Model

    Args:
        features:
        mode:
        params:

    Returns:
        Model function
    """
    network = resnet_v1(
        resnet_depth=50,
        num_classes=params["classes"],
        data_format=params["data_format"],
    )
    return network(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))


def model_fn(features, labels, mode, params):
    """Model function that returns the estimator spec

    Args:
        features: This is the x-arg from the input_fn.
        labels:   This is the y-arg from the input_fn,
                  see e.g. train_input_fn for these two.
        mode:     Either TRAIN, EVAL, or PREDICT
        params:   User-defined hyper-parameters, e.g. learning-rate.
    Returns:
        tf.estimator.EstimatorSpec: Estimator specification
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating model in {} mode".format(mode))

    logits = build_network(features, mode, params)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Softmax output of the neural network.
        y_pred = tf.nn.softmax(logits=logits)

        predictions = {
            "class_ids": y_pred_cls,
            "probabilities": y_pred,
            "logits": logits,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )

    loss = tf.reduce_mean(cross_entropy, name="loss")

    accuracy = tf.metrics.accuracy(labels=labels, predictions=y_pred_cls, name="acc_op")
    metrics = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_hook_list = []
        eval_tensors_log = {"acc": accuracy[1]}
        eval_hook_list.append(
            tf.train.LoggingTensorHook(tensors=eval_tensors_log, every_n_iter=100)
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=metrics,
            loss=loss,
            evaluation_hooks=eval_hook_list,
        )

    optimizer = _get_optimizer(params)

    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    train_hook_list = []
    train_tensors_log = {"loss": loss, "acc": accuracy[1]}
    train_hook_list.append(
        tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=100)
    )

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, training_hooks=train_hook_list
    )


def _get_runconfig(is_distributed=defaults.DISTRIBUTED, save_checkpoints_steps=None):
    if is_distributed:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        return tf.estimator.RunConfig(
            save_checkpoints_steps=save_checkpoints_steps,
            save_checkpoints_secs=None,
            session_config=config,
            log_step_count_steps=100,
        )
    else:
        return tf.estimator.RunConfig(
            save_checkpoints_steps=save_checkpoints_steps,
            save_checkpoints_secs=None,
            log_step_count_steps=100,
        )


def _get_hooks(batch_size, is_distributed=defaults.DISTRIBUTED):
    logger = logging.getLogger(__name__)

    if is_distributed:
        exps_hook = ExamplesPerSecondHook(batch_size * hvd.size())
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        logger.info("Rank: {} Cluster Size {}".format(hvd.rank(), hvd.size()))
        return [bcast_hook, exps_hook]
    else:
        exps_hook = ExamplesPerSecondHook(batch_size)
        return [exps_hook]


def _is_master(is_distributed=defaults.DISTRIBUTED):
    if is_distributed:
        if hvd.rank() == 0:
            return True
        else:
            return False
    else:
        return True


def _log_summary(total_images, batch_size, duration):
    logger = logging.getLogger(__name__)
    images_per_second = total_images / duration
    logger.info("Data length:      {}".format(total_images))
    logger.info("Total duration:   {:.3f}".format(duration))
    logger.info("Total images/sec: {:.3f}".format(images_per_second))
    logger.info(
        "Batch size:       (Per GPU {}: Total {})".format(
            batch_size, hvd.size() * batch_size if defaults.DISTRIBUTED else batch_size
        )
    )
    logger.info(
        "Distributed:      {}".format("True" if defaults.DISTRIBUTED else "False")
    )
    logger.info(
        "Num GPUs:         {:.3f}".format(hvd.size() if defaults.DISTRIBUTED else 1)
    )


def main(
    training_data_path=None,
    validation_data_path=None,
    save_filepath="logs",
    epochs=defaults.EPOCHS,
    batch_size=defaults._BATCHSIZE,
    max_steps=None,
    save_checkpoints_steps=None,
    data_format="channels_last",
    momentum=0.9,
    data_type="tfrecords"
):
    """Run train and evaluation loop

    Args:
        training_data_path: Location of training data
        validation_data_path: Location of validation data
        save_filepath: Location where the checkpoint and events files are saved
        epochs: Number of epochs to run the training for
        batch_size: Number of images to run in a mini-batch
        max_steps: Maximum number of steps to run for training. This will override epochs parameter
        save_checkpoints_steps: Number of steps between checkpoints
        data_format: The axis order of the matrix, channels_last NHWC or channels_first NCHW
        momentum: Momentum term for tf.train.MomentumOptimizer
        data_type: The format that the data is in, valid values are 'images' and 'tfrecords'
    """
    logger = logging.getLogger(__name__)
    if defaults.DISTRIBUTED:
        # Horovod: initialize Horovod.
        hvd.init()
        logger.info("Runnin Distributed")
        logger.info("Num GPUs: {:.3f}".format(hvd.size()))

    logger.info("Tensorflow version {}".format(tf.__version__))
    if training_data_path is None:
        steps=None
        input_function = get_synth_input_fn(
            defaults.DEFAULT_IMAGE_SIZE,
            defaults.DEFAULT_IMAGE_SIZE,
            defaults.NUM_CHANNELS,
            defaults.NUM_CLASSES,
        )
    else:
        total_batches = (defaults.NUM_IMAGES['train'] / batch_size)
        steps = total_batches // hvd.size() if defaults.DISTRIBUTED else total_batches
        input_function = tfrecords.input_fn if "tfrecords" in data_type else images.input_fn
        logger.info(f"Running {steps} steps")

    run_config = _get_runconfig(save_checkpoints_steps=save_checkpoints_steps)
    if (defaults.DISTRIBUTED and hvd.rank() == 0) or not defaults.DISTRIBUTED:
        model_dir = save_filepath
    else:
        model_dir = "."

    params = {
        "learning_rate": defaults.LR,
        "momentum": momentum,
        "classes": defaults.NUM_CLASSES,
        "data_format": data_format,
    }
    logger.info("Creating estimator with params: {}".format(params))
    model = tf.estimator.Estimator(
        model_fn=model_fn, params=params, model_dir=model_dir, config=run_config
    )

    hooks = _get_hooks(batch_size)
    num_gpus = hvd.size() if defaults.DISTRIBUTED else 1

    def train_input_fn():
        return input_function(
            True,
            training_data_path,
            batch_size,
            repetitions=epochs+1, # Repeat the dataset one more than the epochs 
            data_format=data_format,
            num_parallel_batches=4,
            distributed=defaults.DISTRIBUTED
        )

    with Timer(output=logger.info, prefix="Training") as t:
        logger.info("Training...")
        model.train(input_fn=train_input_fn, steps=steps, max_steps=max_steps, hooks=hooks)

    if max_steps is not None:
        total_images = max_steps * batch_size * num_gpus
    else:
        total_images = epochs * defaults.NUM_IMAGES["train"]

    _log_summary(total_images, batch_size, t.elapsed)

    if _is_master() and validation_data_path is not None:

        def validation_input_fn():
            return input_function(
                False,
                validation_data_path,
                batch_size,
                repetitions=1,
                data_format=data_format,
                num_parallel_batches=4,
            )

        with Timer(output=logger.info, prefix="Testing"):
            logger.info("Testing...")
            model.evaluate(input_fn=validation_input_fn)


if __name__ == "__main__":
    logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))
    fire.Fire(main)
