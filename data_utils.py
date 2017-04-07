from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

try:
  import cPickle as pickle
except:
  import pickle

import numpy as np
import scipy.misc

IMAGE_SIZE = 32
TRAIN_VAL_RATIO = 0.85

def to_one_hot(y, num_label):
  """Converts a one-dimensional label array to a two-dimensional one-hot array"""
  return np.eye(num_label)[y]

def load_image(path, preprocess=False, normalize=None):
  """Loads an image from the disk

  After loaded from the disk, the 3-channel image is flattened to a grayscale
  image. If preprocess is set to True, the image is resized to 32x32 using
  bilinear interpolation.

  Args:
    path: path to the image
    preprocess: if set to True, the image is preprocessed. Default is False
    normalize: a tuple of the form (mean, std). Each of these two values is a
               one-dimensional array of the size that matches the size of the
               image. Each pixel of the image is subtracted and divided by the
               corresponding value in each of these arrays. Default is None
  Returns:
    A two-dimensional floating-point array containing grayscale value of each
    pixel of the loaded image.

  """
  img = scipy.misc.imread(path, flatten=True)
  if preprocess:
    img = scipy.misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear')
  if normalize is not None:
    mean, std = normalize
    img = ((img.ravel() - mean)/std).reshape(img.shape)

  return img

class ImageDataset:
  """A class that contains datasets of images

  On construction, this class load images and labels from the specified path.
  As a convention for this project, this path can either be
  'dataset/detectorData' or 'dataset/recognizerData'. Each of these paths
  contains a full training set and a test set.

  The full training set is later split into training set and validation set. On
  splitting, the mean and standard deviation of each feature in the training set
  is recorded as mean_train and std_train. These two arrays are used to
  normalized all training set, validation set and test set.

  Attributes:
    X_train: input of training set
    y_train: label of training set
    X_val: input of validation set
    y_val: label of validation set
    X_test: input of test set
    y_test: label of test set
    mean_train: mean of the training set
    std_train: standard deviation of the training set

  """
  def __init__(self, location_prefix, from_pickle=False):
    self.mean_train = None
    self.std_train  = None
    if from_pickle:
      self.unpickle(location_prefix)
    else:
      self._load_images(location_prefix)

  def _load_images(self, prefix):
    train_label_path  = os.path.join(prefix, 'trainLabels.csv')
    test_label_path   = os.path.join(prefix, 'testLabels.csv')

    self.all_y_train = np.genfromtxt(train_label_path, delimiter=',', usecols=1,
                                     dtype=np.int32)
    self.y_test  = np.genfromtxt(test_label_path, delimiter=',', usecols=1,
                                 dtype=np.int32)

    N_all_train = self.all_y_train.shape[0]
    N_test  = self.y_test.shape[0]

    print('Loading train set...')
    all_X_train = []
    for i in xrange(1, N_all_train + 1):
      img_path = os.path.join(prefix, 'train', str(i) + '.png')
      I = load_image(img_path)
      all_X_train.append(I.ravel())
    print('Finished loading train set')

    self.all_X_train = np.array(all_X_train)

    self._split_train_val()

    print('Loading test set...')
    X_test = []
    for i in xrange(1, N_test + 1):
      img_path = os.path.join(prefix, 'test', str(i) + '.png')
      I = load_image(img_path)
      X_test.append(I.ravel())
    print('Finished loading test set')

    self.X_test = np.array(X_test)

    self.X_train = self.normalize(self.X_train)
    self.X_val   = self.normalize(self.X_val)
    self.X_test  = self.normalize(self.X_test)

  def pickle(self, prefix):
    """Saves the loaded dataset as pickle files

    Args:
      prefix: path to the directory containing the pickle files.

    """
    with open(os.path.join(prefix, 'X_train.pickle'), 'wb') as f:
      pickle.dump(self.all_X_train, f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'wb') as f:
      pickle.dump(self.all_y_train, f)
    with open(os.path.join(prefix, 'X_val.pickle'), 'wb') as f:
      pickle.dump(self.all_X_train, f)
    with open(os.path.join(prefix, 'y_val.pickle'), 'wb') as f:
      pickle.dump(self.all_y_train, f)
    with open(os.path.join(prefix, 'X_test.pickle'), 'wb') as f:
      pickle.dump(self.X_test, f)
    with open(os.path.join(prefix, 'y_test.pickle'), 'wb') as f:
      pickle.dump(self.y_test, f)

  def unpickle(self, prefix):
    """Loads the dataset from pickle files

    Args:
      prefix: path to the directory containing the pickle files.

    """
    with open(os.path.join(prefix, 'X_train.pickle'), 'rb') as f:
      self.all_X_train = pickle.load(f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'rb') as f:
      self.all_y_train = pickle.load(f)
    with open(os.path.join(prefix, 'X_val.pickle'), 'wb') as f:
      pickle.dump(self.all_X_train, f)
    with open(os.path.join(prefix, 'y_val.pickle'), 'wb') as f:
      pickle.dump(self.all_y_train, f)
    with open(os.path.join(prefix, 'X_test.pickle'), 'rb') as f:
      self.X_test = pickle.load(f)
    with open(os.path.join(prefix, 'y_test.pickle'), 'rb') as f:
      self.y_test = pickle.load(f)

    self._split_train_val()

  def _split_train_val(self):
    perm = np.arange(self.all_X_train.shape[0])
    np.random.shuffle(perm)
    self.all_X_train = self.all_X_train[perm]
    self.all_y_train = self.all_y_train[perm]

    train_val_split = int(self.all_y_train.shape[0] * TRAIN_VAL_RATIO)
    self.X_train = self.all_X_train[:train_val_split]
    self.y_train = self.all_y_train[:train_val_split]
    self.X_val   = self.all_X_train[train_val_split:]
    self.y_val   = self.all_y_train[train_val_split:]

    self.mean_train = np.mean(self.X_train, axis=0)
    self.std_train  = np.std(self.X_train, axis=0)

  def _shuffle_train(self):
    perm = np.arange(self.X_train.shape[0])
    np.random.shuffle(perm)
    self.X_train = self.X_train[perm]
    self.y_train = self.y_train[perm]

  def normalize(self, sample):
    if self.mean_train is not None:
      return (sample - self.mean_train) / self.std_train
    return sample

  def save_normalize(self, path):
    """Saves normalization values to a file

    Normalization values are the mean and standard deviation of the training
    set. This is saved as a tuple (mean, std) to a pickle file.

    Args:
      path: path to the pickle file that the normalization is saved to.

    """
    if self.mean_train is not None:
      if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
      with open(path, 'wb') as f:
        pickle.dump((self.mean_train, self.std_train), f)

  def train_batches(self, batch_size):
    """Returns a generator for fetching a batch of training data

    Args:
      batch_size: size of each batch

    """
    self._shuffle_train()
    for start in xrange(0, self.X_train.shape[0], batch_size):
      yield ( self.X_train[start: start + batch_size],
              self.y_train[start: start + batch_size])
