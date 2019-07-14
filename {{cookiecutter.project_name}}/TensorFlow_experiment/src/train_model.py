"""Script to train model using TensorFlow and Horovod

Please complete the necessary functions and assign values to the required variables


For instructions on using TensorFLow see: https://www.tensorflow.org/
For instructions on using Horovod see: https://github.com/horovod/horovod

"""
import logging.config
import fire
import os
import tensorflow as tf

DISTRIBUTED = False
LR = 0.001
MOMENTUM = 0.9
NUM_CLASSES = #Number of classes for your dataset

if DISTRIBUTED:
    import horovod.tensorflow as hvd


def _get_rank():
    if DISTRIBUTED:
        try:
            return hvd.rank()
        except:
            return 0
    else:
        return 0


def _get_optimizer(params, is_distributed=DISTRIBUTED):
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
    """ Build Model

    Args:
        features:
        mode:
        params:

    Returns:
        Model function

    """
    return None


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
    model = build_network(features, mode, params)
    return None


def input_fn():
    """Input function which provides batches for train or eval.

    Returns:
        A dataset that can be used for iteration.
    """
    return None


def _get_runconfig(is_distributed=DISTRIBUTED, save_checkpoints_steps=None):
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


def _get_hooks(is_distributed=DISTRIBUTED):
    logger = logging.getLogger(__name__)
    if is_distributed:
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        logger.info("Rank: {} Cluster Size {}".format(hvd.local_rank(), hvd.size()))
        return [bcast_hook]
    else:
        return []


def main():
    """Train your model
    """
    logger = logging.getLogger(__name__)
    if DISTRIBUTED:
        # Horovod: initialize Horovod.
        hvd.init()
        logger.info("Running Distributed")
        logger.info("Num GPUs: {:.3f}".format(hvd.size()))

    input_function = input_fn

    run_config = _get_runconfig()

    params = {
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "classes": NUM_CLASSES,
    }
    logger.info("Creating estimator with params: {}".format(params))
    model = tf.estimator.Estimator(
        model_fn=model_fn, params=params, config=run_config
    )

    hooks = _get_hooks()

    model.train(input_fn=input_function, hooks=hooks)

    model.evaluate(input_fn=input_function)


if __name__ == "__main__":
    logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))
    fire.Fire(main)
