"""
Trains ResNet50 using CNTK.

It requires the following env variables
AZ_BATCHAI_INPUT_TRAIN
AZ_BATCHAI_OUTPUT_MODEL
 
This code is based on this example:
https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/TrainResNet_ImageNet_Distributed.py
"""

from __future__ import print_function
import os
import cntk as C
import numpy as np

from cntk import input, cross_entropy_with_softmax, classification_error, Trainer, cntk_py
from cntk import data_parallel_distributed_learner, Communicator
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule, UnitType
from cntk.train.training_session import *
from cntk.debugging import *
from cntk.logging import *
from resnet_models import *
import cntk.io.transforms as xforms
import logging

logger = logging.getLogger(__name__)

# model dimensions
_WIDTH = 224
_HEIGHT = 224
_CHANNELS = 3
_LR = 0.001
_EPOCHS = 1
_BATCHSIZE = 64
_MOMENTUM = 0.9
_NUMCLASSES = 1000
_MODELNAME = 'ResNet_ImageNet.model'
_NUMQUANTIZEDBITS = 32
_WD = 0.0001
_NUMIMAGES = 1281167


def _get_progress_printer():
    pp = ProgressPrinter(
        freq=100,
        tag='Training',
        log_to_file=None,
        rank=Communicator.rank(),
        gen_heartbeat=False,
        num_epochs=_EPOCHS)
    return pp


def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist." %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomarea',
                        area_ratio=(0.08, 1.0),
                        aspect_ratio=(0.75, 1.3333),
                        jitter_type='uniratio')
        ]
    else:
        transforms += [
            # test has no jitter
            C.io.transforms.crop(crop_type='center', side_ratio=0.875)
        ]

    transforms += [
        xforms.scale(width=_WIDTH, height=_HEIGHT,
                     channels=_CHANNELS, interpolations='cubic'),
        xforms.mean(mean_file)
    ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            # 1st col in mapfile referred to as 'image'
            features=C.io.StreamDef(field='image', transforms=transforms),
            labels=C.io.StreamDef(field='label', shape=_NUMCLASSES))),     # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)


def model_fn():
    # Input variables denoting the features and label data
    graph_input = C.input_variable((_CHANNELS, _HEIGHT, _WIDTH))
    graph_label = C.input_variable((_NUMCLASSES))

    with C.default_options(dtype=np.float32):
        stride1x1 = (1, 1)
        stride3x3 = (2, 2)

        # create model, and configure learning parameters for ResNet50
        z = create_imagenet_model_bottleneck(graph_input, [2, 3, 5, 2],
                                             _NUMCLASSES, stride1x1, stride3x3)

        # loss and metric
        ce = cross_entropy_with_softmax(z, graph_label)
        errs = classification_error(z, graph_label, topN=1)

    return {
        'name': 'resnet50',
        'feature': graph_input,
        'label': graph_label,
        'ce': ce,
        'errs': errs,
        'output': z
    }


# Create trainer
def create_trainer(network, minibatch_size, epoch_size,
                   learning_rate, momentum, l2_reg_weight,
                   num_quantization_bits):
    lr_per_mb = [learning_rate]

    # Set learning parameters
    lr_schedule = learning_rate_schedule(
        lr_per_mb, epoch_size=epoch_size, unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(momentum)
    local_learner = momentum_sgd(network['output'].parameters,
                                 lr_schedule,
                                 mm_schedule,
                                 l2_regularization_weight=l2_reg_weight)

    # learner object
    learner = data_parallel_distributed_learner(
        local_learner,
        num_quantization_bits=num_quantization_bits,
        distributed_after=0)

    # logger
    progress_printer = _get_progress_printer()

    return Trainer(network['output'], (network['ce'], network['errs']), learner, progress_printer)


def train_and_test(network, trainer, train_source, test_source, minibatch_size,
                   epoch_size, model_path):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    training_session(
        trainer=trainer,
        mb_source=train_source,
        mb_size=minibatch_size,
        model_inputs_to_streams=input_map,
        progress_frequency=epoch_size,
        checkpoint_config=CheckpointConfig(frequency=epoch_size,
                                           filename=os.path.join(
                                               model_path, _MODELNAME),
                                           restore=False)  # ,
        # test_config=TestConfig(test_source, minibatch_size)
    ).train()


def main():
    model_path = os.getenv('AZ_BATCHAI_OUTPUT_MODEL')
    data_path = os.getenv('AZ_BATCHAI_INPUT_TRAIN')
    logger.info("model_path: {}".format(model_path))
    logger.info("data_path: {}".format(data_path))
    mean_data = os.path.join(data_path, 'ImageNet1K_mean.xml')
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')
    logger.info("AZ_BATCHAI_NUM_GPUS={}".format(os.getenv('AZ_BATCHAI_NUM_GPUS')))
    logger.info("AZ_BATCHAI_WORKER_HOSTS={}".format(os.getenv('AZ_BATCHAI_WORKER_HOSTS')))
    
    #set_computation_network_trace_level(0)
    logger.info("mean_data: {}".format(mean_data))
    #logger.info("communicator num_workers {}".format(Communicator.num_workers()))
    #logger.info(type(Communicator.num_workers()))
    #minibatch_size = _BATCHSIZE * Communicator.num_workers()
    minibatch_size = _BATCHSIZE*32

    logger.info("Creating model...")
    network = model_fn()
    trainer = create_trainer(network,
                             minibatch_size,
                             _NUMIMAGES,
                             learning_rate=_LR,
                             momentum=_MOMENTUM,
                             l2_reg_weight=_WD,
                             num_quantization_bits=_NUMQUANTIZEDBITS)
    
    logger.info('Creating data sources...')
    train_source = create_image_mb_source(
        train_data, mean_data, train=True, total_number_of_samples=_EPOCHS*_NUMIMAGES)
    test_source = create_image_mb_source(
        test_data, mean_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    
    logger.info('Training...')
    train_and_test(network, trainer, train_source, test_source,
                   minibatch_size, _NUMIMAGES, model_path)

    # Must call MPI finalize when process exit without exceptions
    Communicator.finalize()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting routine")
    main()
    logger.info("Routine finished")
    
