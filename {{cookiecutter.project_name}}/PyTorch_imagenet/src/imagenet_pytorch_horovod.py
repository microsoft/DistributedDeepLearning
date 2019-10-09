""" Trains ResNet50 in PyTorch using Horovod.
"""
import logging
import logging.config
import os
import shutil
from os import path

import fire
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torchvision.models as models
from azureml.core.run import Run
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from timer import Timer


def _str_to_bool(in_str):
    if "t" in in_str.lower():
        return True
    else:
        return False


_WIDTH = 224
_HEIGHT = 224
_CHANNELS = 3
_LR = 0.001
_EPOCHS = os.getenv("EPOCHS", 5)
_BATCHSIZE = 64
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_SD = [0.229, 0.224, 0.225]
_SEED = 42

# Settings from https://arxiv.org/abs/1706.02677.
_WARMUP_EPOCHS = 5
_WEIGHT_DECAY = 0.00005

_DATA_LENGTH = int(
    os.getenv("FAKE_DATA_LENGTH", 1281167)  # 1281167
)  # How much fake data to simulate, default to size of imagenet dataset
_DISTRIBUTED = _str_to_bool(os.getenv("DISTRIBUTED", "False"))

if _DISTRIBUTED:
    import horovod.torch as hvd

    hvd.init()


def _get_rank():
    if _DISTRIBUTED:
        try:
            return hvd.rank()
        except:
            return 0
    else:
        return 0


def _append_path_to(data_path, data_series):
    return data_series.apply(lambda x: path.join(data_path, x))


def _create_data(batch_size, num_batches, dim, channels, seed=42):
    np.random.seed(seed)
    return np.random.rand(batch_size * num_batches, channels, dim[0], dim[1]).astype(
        np.float32
    )


def _create_labels(batch_size, num_batches, n_classes):
    return np.random.choice(n_classes, batch_size * num_batches)


class FakeData(Dataset):
    def __init__(
        self,
        batch_size=32,
        num_batches=20,
        dim=(224, 224),
        n_channels=3,
        n_classes=10,
        length=_DATA_LENGTH,
        data_transform=None,
    ):
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_batches = num_batches
        self._data = _create_data(
            batch_size, self.num_batches, self.dim, self.n_channels
        )
        self._labels = _create_labels(batch_size, self.num_batches, self.n_classes)
        self.translation_index = np.random.choice(len(self._labels), length)
        self._length = length

        self._data_transform = data_transform
        logger = logging.getLogger(__name__)
        logger.info(
            "Creating fake data {} labels and {} images".format(
                n_classes, len(self._data)
            )
        )

    def __getitem__(self, idx):
        logger = logging.getLogger(__name__)
        logger.debug("Retrieving samples")
        logger.debug(str(idx))
        tr_index_array = self.translation_index[idx]

        if self._data_transform is not None:
            data = self._data_transform(self._data[tr_index_array])
        else:
            data = self._data[tr_index_array]

        return data, self._labels[tr_index_array]

    def __len__(self):
        return self._length


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def avg(self):
        return self._sum / self._count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, base_lr, warmup_epochs, epoch):
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    msg = " duration({})  loss:{} total-samples: {}"
    t = Timer()
    t.start()
    for i, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, base_lr, warmup_epochs, train_loader, epoch, i)
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # compute output
        output = model(data)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        if i % 100 == 0:
            t.stop()
            batch_time.update(t.elapsed, n=100)
            logger.info(msg.format(t.elapsed, loss.item(), i * len(data)))
            t.start()

    return {"acc": top1.avg, "loss": losses.avg, "batch_time": batch_time.avg}


def validate(val_loader, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger = logging.getLogger(__name__)
    msg = " duration({})  loss:{} total-samples: {}"
    t = Timer()
    t.start()
    for i, (data, target) in enumerate(val_loader):
        logger.debug("bug")
        data, target = (
            data.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
        # compute output
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        if i % 100 == 0:
            logger.info(msg.format(t.elapsed, loss.item(), i * len(data)))
            t.start()
    return {"acc": top1.avg, "loss": losses.avg}


def _log_summary(data_length, duration, batch_size):
    logger = logging.getLogger(__name__)
    images_per_second = data_length / duration
    logger.info("Data length:      {}".format(data_length))
    logger.info("Total duration:   {:.3f}".format(duration))
    logger.info("Total images/sec: {:.3f}".format(images_per_second))
    logger.info(
        "Batch size:       (Per GPU {}: Total {})".format(
            batch_size, hvd.size() * batch_size if _DISTRIBUTED else batch_size
        )
    )
    logger.info("Distributed:      {}".format("True" if _DISTRIBUTED else "False"))
    logger.info("Num GPUs:         {:.3f}".format(hvd.size() if _DISTRIBUTED else 1))


def _get_sampler(dataset, is_distributed=_DISTRIBUTED):
    if is_distributed:
        return torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
    else:
        return torch.utils.data.sampler.RandomSampler(dataset)


def save_checkpoint(model, optimizer, filepath):
    if hvd.rank() == 0:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, filepath)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(
    optimizer, base_lr, warmup_epochs, data_loader, epoch, batch_idx
):
    logger = logging.getLogger(__name__)
    size = hvd.size() if _DISTRIBUTED else 1
    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / len(data_loader)
        lr_adj = 1.0 / size * (epoch * (size - 1) / warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.0
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        new_lr = base_lr * size * lr_adj
        if param_group["lr"]!=new_lr:
            param_group["lr"] = new_lr
            if _get_rank()==0:
                logger.info(f"setting lr to {param_group['lr']}")


def main(
    training_data_path=None,
    validation_data_path=None,
    use_gpu=False,
    save_filepath=None,
    model="resnet50",
    epochs=_EPOCHS,
    batch_size=_BATCHSIZE,
    fp16_allreduce=False,
    base_lr=0.0125,
    warmup_epochs=5,
):
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info(f"Running on {device}")
    if _DISTRIBUTED:
        # Horovod: initialize Horovod.

        logger.info("Running Distributed")
        torch.manual_seed(_SEED)
        if use_gpu:
            # Horovod: pin GPU to local rank.
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(_SEED)

    logger.info("PyTorch version {}".format(torch.__version__))

    # Horovod: write TensorBoard logs on first worker.
    if (_DISTRIBUTED and hvd.rank() == 0) or not _DISTRIBUTED:
        run = Run.get_context()
        run.tag("model", value=model)

        logs_dir = os.path.join(os.curdir, "logs")
        if os.path.exists(logs_dir):
            logger.debug(f"Log directory {logs_dir} found | Deleting")
            shutil.rmtree(logs_dir)
        summary_writer = SummaryWriter(logdir=logs_dir)

    if training_data_path is None:
        logger.info("Setting up fake loaders")
        train_dataset = FakeData(n_classes=1000, data_transform=torch.FloatTensor)
        validation_dataset = None
    else:
        normalize = transforms.Normalize(_RGB_MEAN, _RGB_SD)
        logger.info("Setting up loaders")
        logger.info(f"Loading training from {training_data_path}")
        train_dataset = datasets.ImageFolder(
            training_data_path,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(_WIDTH),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if validation_data_path is not None:
            logger.info(f"Loading validation from {validation_data_path}")
            validation_dataset = datasets.ImageFolder(
                validation_data_path,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

    train_sampler = _get_sampler(train_dataset)
    kwargs = {"num_workers": 5, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    if validation_data_path is not None:
        val_sampler = _get_sampler(validation_dataset)
        val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, sampler=val_sampler, **kwargs
        )

    # Autotune
    cudnn.benchmark = True

    logger.info("Loading model")

    # Load symbol
    model = models.__dict__[model](pretrained=False)

    # model.to(device)
    if use_gpu:
        # Move model to GPU.
        model.cuda()

    # # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    num_gpus = hvd.size() if _DISTRIBUTED else 1
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=_LR * num_gpus, momentum=0.9)
    if _DISTRIBUTED:

        compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
        )

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = F.cross_entropy

    # Main training-loop
    logger.info("Training ...")
    for epoch in range(epochs):
        with Timer(output=logger.info, prefix=f"Training epoch {epoch} ") as t:
            model.train()
            if _DISTRIBUTED:
                train_sampler.set_epoch(epoch)
            metrics = train(
                train_loader, model, criterion, optimizer, base_lr, warmup_epochs, epoch
            )

            if (_DISTRIBUTED and hvd.rank() == 0) or not _DISTRIBUTED:
                run.log_row("Training metrics", epoch=epoch, **metrics)
                summary_writer.add_scalar("Train/Loss", metrics["loss"], epoch)
                summary_writer.add_scalar("Train/Acc", metrics["acc"], epoch)
                summary_writer.add_scalar("Train/BatchTime", metrics["batch_time"], epoch)

        if validation_data_path is not None:
            model.eval()
            metrics = validate(val_loader, model, criterion, device)
            if (_DISTRIBUTED and hvd.rank() == 0) or not _DISTRIBUTED:
                run.log_row("Validation metrics", epoch=epoch, **metrics)
                summary_writer.add_scalar("Validation/Loss", metrics["loss"], epoch)
                summary_writer.add_scalar("Validation/Acc", metrics["acc"], epoch)

        if save_filepath is not None:
            save_checkpoint(model, optimizer, save_filepath)

    _log_summary(epochs * len(train_dataset), t.elapsed, batch_size)


if __name__ == "__main__":
    logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))
    fire.Fire(main)
