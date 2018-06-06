import numpy as np
import keras
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_data(batch_size, num_batches, dim, channels, seed=42):
    np.random.seed(42)
    return np.random.rand(batch_size * num_batches,
                          dim[0],
                          dim[1],
                          channels).astype(np.float32)


def _create_labels(batch_size, num_batches, n_classes):
    return np.random.choice(n_classes, batch_size * num_batches)

#
# class FakeDataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#
#     def __init__(self,
#                  batch_size=32,
#                  num_batches=20,
#                  dim=(224, 224),
#                  n_channels=3,
#                  n_classes=10,
#                  length=1000,
#                  shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.num_batches = num_batches
#         self._data = _create_data(self.batch_size, self.num_batches, self.dim, self.n_channels)
#         self._labels = _create_labels(self.batch_size, self.num_batches, self.n_classes)
#         self._indexes = np.arange(len(self._labels))
#         self._length=length
#         self.batch_index=0
#         self.on_epoch_end()
#
#     def __len__(self):
#         return self._length
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self._indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         # Generate data
#         X, y = self._data_generation(indexes)
#
#         return X, y
#
#     def __iter__(self):  # pylint: disable=non-iterator-returned
#
#         # Needed if we want to do something like:
#         # for x, y in data_gen.flow(...):
#         return self
#
#     def __next__(self, *args, **kwargs):
#         return self.next(*args, **kwargs)
#
#     def reset(self):
#         self.batch_index = 0
#
#     def next(self):
#         """For python 2.x.
#         Returns:
#             The next batch.
#         """
#         # Keeps under lock only the mechanism which advances
#         # the indexing of each batch.
#         with self.lock:
#             index_array = self.batch_index
#             self.batch_index+=1
#             if self.batch_index>=len(self._labels):
#                 self.reset()
#         # The transformation of images is not under thread lock
#         # so it can be done in parallel
#
#         return self[index_array]
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         if self.shuffle == True:
#             np.random.shuffle(self._indexes)
#
#     def _data_generation(self, indexes):
#         'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
#         return self._data[indexes], keras.utils.to_categorical(self._labels[indexes], num_classes=self.n_classes)




class FakeDataGenerator(keras.preprocessing.image.Iterator):

    def __init__(self,
                 batch_size=32,
                 num_batches=20,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=10,
                 length=1000,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.num_batches = num_batches
        self._data = _create_data(self.batch_size, self.num_batches, self.dim, self.n_channels)
        self._labels = _create_labels(self.batch_size, self.num_batches, self.n_classes)
        self._indexes = np.arange(len(self._labels))
        self._length=length
        self.batch_index=0

    def _get_batches_of_transformed_samples(self, index_array):
        logger.info('Retrieving samples')
        logger.info(str(index_array))
        return self._data[index_array], keras.utils.to_categorical(self._labels[index_array], num_classes=self.n_classes)