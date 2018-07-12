"""
Trains ResNet50 in Keras using Horovod.

It requires the following env variables
AZ_BATCHAI_INPUT_TRAIN
AZ_BATCHAI_INPUT_TEST
AZ_BATCHAI_OUTPUT_MODEL
AZ_BATCHAI_JOB_TEMP_DIR
"""
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from timer import Timer
import numpy as np
import os
from PIL import Image

import torch.optim as optim
from torchvision import transforms
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision.models as models
from os import path
import pandas as pd
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F

def _str_to_bool(in_str):
    if 't' in in_str.lower():
        return True
    else:
        return False

_WIDTH = 224
_HEIGHT = 224
_CHANNELS = 3
_LR = 0.001
_EPOCHS = 1
_BATCHSIZE = 64
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_SD = [0.229, 0.224, 0.225]
_SEED=42

# Settings from https://arxiv.org/abs/1706.02677.
_WARMUP_EPOCHS = 5
_WEIGHT_DECAY = 0.00005

_FAKE = _str_to_bool(os.getenv('FAKE', 'False'))
_DATA_LENGTH = int(os.getenv('FAKE_DATA_LENGTH', 1281167)) # How much fake data to simulate, default to size of imagenet dataset




_DISTRIBUTED = _str_to_bool(os.getenv('DISTRIBUTED', 'False'))

if _DISTRIBUTED:
    import horovod.torch as hvd


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
    # File-path
    train_X = train_df['filenames'].values
    validation_X = validation_df['filenames'].values
    # One-hot encoded labels for torch
    train_labels = train_df[['num_id']].values.ravel()
    validation_labels = validation_df[['num_id']].values.ravel()
    # Index starts from 0
    train_labels -= 1
    validation_labels -= 1
    return train_X, train_labels, validation_X, validation_labels


class ImageNet(Dataset):
    def __init__(self, img_locs, img_labels, transform=None):
        self.img_locs, self.labels = img_locs, img_labels
        self.transform = transform
        logger.info("Loaded {} labels and {} images".format(len(self.labels), len(self.img_locs)))

    def __getitem__(self, idx):
        im_file = self.img_locs[idx]
        label = self.labels[idx]
        with open(im_file, 'rb') as f:
            im_rgb = Image.open(f)
            # Make sure 3-channel (RGB)
            im_rgb = im_rgb.convert('RGB')
            if self.transform is not None:
                im_rgb = self.transform(im_rgb)
            return im_rgb, label

    def __len__(self):
        return len(self.img_locs)


def _get_logger():
    return logging.getLogger(__name__)

def _create_data(batch_size, num_batches, dim, channels, seed=42):
    np.random.seed(seed)
    return np.random.rand(batch_size * num_batches,
                          channels,
                          dim[0],
                          dim[1]).astype(np.float32)


def _create_labels(batch_size, num_batches, n_classes):
    return np.random.choice(n_classes, batch_size * num_batches)



class FakeData(Dataset):
    def __init__(self,
                 batch_size=32,
                 num_batches=20,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=10,
                 length=_DATA_LENGTH,
                 seed=42,
                 data_transform=None,
                 label_transform=None):
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_batches = num_batches
        self._data = _create_data(batch_size, self.num_batches, self.dim, self.n_channels)
        self._labels = _create_labels(batch_size, self.num_batches, self.n_classes)
        self.translation_index = np.random.choice(len(self._labels), length)
        self._length=length
        # self._data = np.random.rand(length, 3, 224, 224).astype(np.float32)
        # self._labels = np.random.rand(length, num_classes).astype(np.float32)

        self._data_transform = data_transform
        self._label_transform = label_transform
        logger.info("Creating fake data {} labels and {} images".format(n_classes, len(self._data)))

    def __getitem__(self, idx):
        logger = _get_logger()
        logger.debug('Retrieving samples')
        logger.debug(str(idx))
        tr_index_array = self.translation_index[idx]
        logger.debug('*****')
        logger.debug(self._data.shape)
        logger.debug(self._data[tr_index_array].shape)

        if self._data_transform is not None:
            data=self._data_transform(self._data[tr_index_array])
        else:
            data=self._data[tr_index_array]

        if self._label_transform is not None:
            label=self._label_transform([self._labels[tr_index_array]])
        else:
            label=[self._labels[tr_index_array]]

        logger.debug(data.shape)
        return data, label

    def __len__(self):
        return self._length


def _is_master(is_distributed=_DISTRIBUTED):
    if is_distributed:
        if hvd.rank() == 0:
            return True
        else:
            return False
    else:
        return True


def train(train_loader, model, criterion, optimizer, epoch):
    logger.info("Training ...")
    t=Timer()
    t.__enter__()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        print(data.shape)
        # target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # compute output
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = criterion(output, target)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            msg = 'Train Epoch: {}   duration({})  loss:{} total-samples: {}'
            logger.info(msg.format(epoch, t.elapsed, loss.data[0], i * len(data)))
            t.__enter__()


def _log_summary(data_length, duration):
    images_per_second = data_length / duration
    logger.info('Data length:      {}'.format(data_length))
    logger.info('Total duration:   {:.3f}'.format(duration))
    logger.info('Total images/sec: {:.3f}'.format(images_per_second))
    logger.info('Batch size:       (Per GPU {}: Total {})'.format(_BATCHSIZE, hvd.size()*_BATCHSIZE if _DISTRIBUTED else _BATCHSIZE))
    logger.info('Distributed:      {}'.format('True' if _DISTRIBUTED else 'False'))
    logger.info('Num GPUs:         {:.3f}'.format(hvd.size() if _DISTRIBUTED else 1))
    logger.info('Dataset:          {}'.format('Synthetic' if _FAKE else 'Imagenet'))



def main():
    if _DISTRIBUTED:
        # Horovod: initialize Horovod.
        logger.info("Runnin Distributed")
        hvd.init()
        torch.manual_seed(_SEED)
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(_SEED)

    logger.info("PyTorch version {}".format(torch.__version__))

    if _FAKE:
        logger.info("Setting up fake loaders")
        train_dataset = FakeData(n_classes=1000, data_transform=torch.FloatTensor, label_transform=torch.LongTensor)
    else:
        normalize = transforms.Normalize(_RGB_MEAN, _RGB_SD)

        train_X, train_y, valid_X, valid_y = _create_data_fn(os.getenv('AZ_BATCHAI_INPUT_TRAIN'), os.getenv('AZ_BATCHAI_INPUT_TEST'))

        logger.info("Setting up loaders")
        train_dataset = ImageNet(
            train_X,
            train_y,
            transforms.Compose([
                transforms.RandomResizedCrop(_WIDTH),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=_BATCHSIZE, sampler=train_sampler, **kwargs)

    # Autotune
    cudnn.benchmark = True

    logger.info("Loading model")
    # Load symbol
    model = models.__dict__['resnet50'](pretrained=False)

    model.cuda()

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=_LR * hvd.size(),
                          momentum=0.9)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())

    criterion=None
    # Main training-loop
    for epoch in range(_EPOCHS):
        with Timer(output=logger.info, prefix="Training") as t:
            model.train()
            train_sampler.set_epoch(epoch)
            train(train_loader, model, criterion, optimizer, epoch)

        _log_summary(len(train_dataset), t.elapsed)

if __name__ == '__main__':
    main()
