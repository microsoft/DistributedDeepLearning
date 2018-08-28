import argparse
import logging
import os
from os import path
import numpy as np
import pandas as pd
import multiprocessing
from toolz import pipe
from timer import Timer
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.distributed as dist
import torch.utils.data.distributed

print("PyTorch: ", torch.__version__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Distributed training settings
parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--rank', default=-1, type=int, help='rank of the worker')

_WIDTH = 224
_HEIGHT = 224
_LR = 0.001
_EPOCHS = 1
_BATCHSIZE = 64*4
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_SD = [0.229, 0.224, 0.225]
args = parser.parse_args()

def _str_to_bool(in_str):
    if 't' in in_str.lower():
        return True
    else:
        return False

_FAKE = _str_to_bool(os.getenv('FAKE', 'True'))
_DATA_LENGTH = int(os.getenv('FAKE_DATA_LENGTH', 1281167)) # How much fake data to simulate, default to size of imagenet dataset

#_DISTRIBUTED = _str_to_bool(os.getenv('DISTRIBUTED', 'False'))
_DISTRIBUTED = True
#_CPU_COUNT = multiprocessing.cpu_count()
_CPU_COUNT = 8
logger.info("Distributed mode: ", _DISTRIBUTED)
logger.info("CPU Count: ", _CPU_COUNT)


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
    def __init__(self, img_locs, img_labels,  transform=None):
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
  

class FakeData(Dataset):
    def __init__(self,
                 batch_size=32,
                 num_batches=20,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=10,
                 length=_DATA_LENGTH,
                 seed=42,
                 data_transform=None):
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_batches = num_batches
        self._data = _create_data(batch_size, self.num_batches, self.dim, self.n_channels)
        self._labels = _create_labels(batch_size, self.num_batches, self.n_classes)
        self.translation_index = np.random.choice(len(self._labels), length)
        self._length=length

        self._data_transform = data_transform
        #logger = _get_logger()
        logger.info("Creating fake data {} labels and {} images".format(n_classes, len(self._data)))

    def __getitem__(self, idx):
        #logger = _get_logger()
        logger.debug('Retrieving samples')
        logger.debug(str(idx))
        tr_index_array = self.translation_index[idx]

        if self._data_transform is not None:
            data=self._data_transform(self._data[tr_index_array])
        else:
            data=self._data[tr_index_array]

        return data, self._labels[tr_index_array]

    def __len__(self):
        return self._length


def _create_data(batch_size, num_batches, dim, channels, seed=42):
    np.random.seed(seed)
    return np.random.rand(batch_size * num_batches,
                          channels,
                          dim[0],
                          dim[1]).astype(np.float32)


def _create_labels(batch_size, num_batches, n_classes):
    return np.random.choice(n_classes, batch_size * num_batches)

    
def train(train_loader, model, criterion, optimizer, epoch):
    logger.info("Training ...")
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
def validate(val_loader, model, criterion):
    logger.info("Validating ...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    logger.info('Top-1 Accuracy: %.2f %%' % (100 * correct / total))
    

def main():
    # Autotune
    cudnn.benchmark = True 
    # Load symbol
    model = models.__dict__['resnet50'](pretrained=False)
    if _DISTRIBUTED:
        logger.info('Running in distributed mode')
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size, 
            rank=args.rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()
    # Optimisers
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
    # Data-sets
    if _FAKE:
        logger.info("Setting up fake loaders")
        train_dataset = FakeData(n_classes=1000, data_transform=torch.FloatTensor)
    else:
        normalize = transforms.Normalize(_RGB_MEAN, _RGB_SD)
        train_X, train_y, valid_X, valid_y = _create_data_fn(os.getenv('AZ_BATCHAI_INPUT_TRAIN'),
                                                             os.getenv('AZ_BATCHAI_INPUT_TEST'))
        train_dataset = ImageNet(
                train_X,
                train_y,
                transforms.Compose([
                    transforms.RandomResizedCrop(_WIDTH),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize]))


    if _DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    # Data-loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=_BATCHSIZE, shuffle=(train_sampler is None), num_workers=_CPU_COUNT, sampler=train_sampler)

    #val_loader = torch.utils.data.DataLoader(
    #    ImageNet(
    #        valid_X,
    #        valid_y,
    #        transforms.Compose([
    #            transforms.Resize(256),
    #            transforms.CenterCrop(_WIDTH),
    #            transforms.ToTensor(),
    #            normalize])), batch_size=_BATCHSIZE, shuffle=False, 
    #    num_workers=_CPU_COUNT)

    # Main training-loop
    for epoch in range(_EPOCHS):
        if _DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        # Train
        with Timer(output=logger.info, prefix="Training"):
            train(train_loader, model, criterion, optimizer, epoch)
        # Validate
        #with Timer(output=logger.info, prefix="Testing"):
        #    validate(val_loader, model, criterion)
          
    print("Finished")
          
if __name__ == '__main__':
    print("Pytorch")
    main()