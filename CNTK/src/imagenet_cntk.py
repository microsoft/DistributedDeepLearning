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
import numpy as np
import cntk as C
from cntk import input, cross_entropy_with_softmax, classification_error, Trainer, cntk_py
from cntk import data_parallel_distributed_learner, Communicator
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule, UnitType
from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData
from cntk.train.training_session import *
from cntk.debugging import *
from cntk.logging import *
import cntk.io.transforms as xforms
from resnet_models import *
from sklearn.preprocessing import OneHotEncoder

import logging


logger = logging.getLogger(__name__)


def _str_to_bool(in_str):
    if 't' in in_str.lower():
        return True
    else:
        return False


# model dimensions
_WIDTH = 224
_HEIGHT = 224
_CHANNELS = 3
_LR = 0.001
_EPOCHS = os.getenv('EPOCHS', 1)
_BATCHSIZE = 64
_MOMENTUM = 0.9
_NUMCLASSES = 1000
_MODELNAME = 'ResNet_ImageNet.model'
_NUMQUANTIZEDBITS = 32
_WD = 0.0001


_FAKE = _str_to_bool(os.getenv('FAKE', 'False'))
# How much fake data to simulate, default to size of imagenet dataset
_DATA_LENGTH = int(os.getenv('FAKE_DATA_LENGTH', 1281167))
_DISTRIBUTED = _str_to_bool(os.getenv('DISTRIBUTED', 'False'))


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


class FakeDataSource(UserMinibatchSource):
    """Fake data source
    https://cntk.ai/pythondocs/Manual_How_to_create_user_minibatch_sources.html
    """

    def __init__(self, total_n_images, dim, channels, n_classes, seed=42):
        self.dim = dim
        self.total_n_images = total_n_images
        self.channels = channels
        self.n_classes = n_classes
        self.seed = seed
        self.fsi = StreamInformation(name='features', stream_id=0, storage_format='dense',
                                     dtype=np.float32, shape=(self.channels, self.dim[0], self.dim[0],))
        self.lsi = StreamInformation(
            name='labels', stream_id=1, storage_format='dense', dtype=np.float32, shape=(self.n_classes,))
        self.sample_count = 0
        self.next_seq_idx = 0
        super(FakeDataSource, self).__init__()

    def stream_infos(self):
        """
        Override the stream_infos method of the base UserMinibatchSource class
        to provide stream meta information.
        """
        return [self.fsi, self.lsi]

    def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
        """
        Override the next_minibatch method of the base UserMinibatchSource class
        to provide minibatch data.
        """
        np.random.seed(self.seed)
        x = np.random.rand(num_samples, self.channels,
                           self.dim[0], self.dim[1]).astype(np.float32)
        y = np.random.choice(self.n_classes, num_samples)
        y = np.expand_dims(y, axis=-1)
        enc = OneHotEncoder(n_values=self.n_classes, dtype=np.float32,
                            categorical_features='all')
        fit = enc.fit(y)
        y = fit.transform(y).toarray()
        if self.sample_count + num_samples <= self.total_n_images:
            self.sample_count += num_samples
            self.next_seq_idx += num_samples
            feature_data = C.Value(batch=x, device=device)
            label_data = C.Value(batch=y, device=device)
            res = {
                self.fsi: MinibatchData(feature_data, num_samples, num_samples, False),
                self.lsi: MinibatchData(label_data, num_samples, num_samples, False)
            }
        else:
            res = {}

        return res

    def get_checkpoint_state(self):
        return {'next_seq_idx': self.next_seq_idx}

    def restore_from_checkpoint(self, state):
        self.next_seq_idx = state['next_seq_idx']


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
    if _DISTRIBUTED:
        learner = data_parallel_distributed_learner(
            local_learner,
            num_quantization_bits=num_quantization_bits,
            distributed_after=0)
    else:
        learner = local_learner

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
    if _DISTRIBUTED:
        start_profiler(sync_gpu=True)

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
    if _DISTRIBUTED:
        stop_profiler()


def main():
    model_path = os.getenv('AZ_BATCHAI_OUTPUT_MODEL')

    if _DISTRIBUTED:
        minibatch_size = _BATCHSIZE * Communicator.num_workers()
    else:
        minibatch_size = _BATCHSIZE

    logger.info("Creating model ...")
    network = model_fn()

    logger.info("Creating trainer ...")
    trainer = create_trainer(network,
                             minibatch_size,
                             _DATA_LENGTH,
                             learning_rate=_LR,
                             momentum=_MOMENTUM,
                             l2_reg_weight=_WD,
                             num_quantization_bits=_NUMQUANTIZEDBITS)

    logger.info('Creating data sources ...')
    if _FAKE:
        train_source = FakeDataSource(total_n_images=_DATA_LENGTH,
                                      dim=(_HEIGHT, _WIDTH),
                                      channels=_CHANNELS,
                                      n_classes=_NUMCLASSES)
        test_source = None
    else:
        data_path = os.getenv('AZ_BATCHAI_INPUT_TRAIN')
        logger.info("model_path: {}".format(model_path))
        logger.info("data_path: {}".format(data_path))

        mean_data = os.path.join(data_path, 'ImageNet1K_mean.xml')
        train_data = os.path.join(data_path, 'train_map.txt')
        test_data = os.path.join(data_path, 'val_map.txt')
        train_source = create_image_mb_source(
            train_data, mean_data, train=True, total_number_of_samples=_EPOCHS*_DATA_LENGTH)
        test_source = create_image_mb_source(
            test_data, mean_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)

    logger.info('Training...')
    train_and_test(network, trainer, train_source, test_source,
                   minibatch_size, _DATA_LENGTH, model_path)

    if _DISTRIBUTED:
        # Must call MPI finalize when process exit without exceptions
        Communicator.finalize()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting routine. Distributed mode={}".format(_DISTRIBUTED))
    main()
    logger.info("Routine finished")
