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

class ImageDataset:

  def __init__(self, location_prefix):
    self._load_images(location_prefix)

  # TODO: preprocess data after loading
  def _load_images(self, prefix):
    train_label_path  = os.path.join(prefix, 'trainLabels.csv')
    test_label_path   = os.path.join(prefix, 'testLabels.csv')

    self.all_y_train = np.genfromtxt(train_label_path, delimiter=',', usecols=1, dtype=np.int32)
    self.y_test  = np.genfromtxt(test_label_path, delimiter=',', usecols=1, dtype=np.int32)

    # Encode as one-hot vectors
    num_label = np.max(self.all_y_train)

    self.all_y_train = np.eye(num_label)[self.all_y_train - 1]
    self.y_test = np.eye(num_label)[self.y_test - 1]

    N_all_train = self.all_y_train.shape[0]
    N_test  = self.y_test.shape[0]

    print('Loading train set...')
    all_X_train = []
    for i in xrange(1, N_all_train + 1):
      img_path = os.path.join(prefix, 'train', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      all_X_train.append(I.ravel())
    print('Finished loading train set')

    self.all_X_train = np.array(all_X_train)

    self._split_train_val()

    print('Loading test set...')
    X_test = []
    for i in xrange(1, N_test + 1):
      img_path = os.path.join(prefix, 'test', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      X_test.append(I.ravel())
    print('Finished loading test set')

    self.X_test = np.array(X_test)

  def pickle(self, prefix):
    with open(os.path.join(prefix, 'X_train.pickle'), 'wb') as f:
      pickle.dump(self.all_X_train, f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'wb') as f:
      pickle.dump(self.all_y_train, f)
    with open(os.path.join(prefix, 'X_test.pickle'), 'wb') as f:
      pickle.dump(self.X_test, f)
    with open(os.path.join(prefix, 'y_test.pickle'), 'wb') as f:
      pickle.dump(self.y_test, f)

  def unpickle(self, prefix):
    with open(os.path.join(prefix, 'X_train.pickle'), 'rb') as f:
      self.all_X_train = pickle.load(f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'rb') as f:
      self.all_y_train = pickle.load(f)
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

    train_val_split = int(self.all_y_train.shape[0] * 0.85)
    self.X_train = self.all_X_train[:train_val_split]
    self.y_train = self.all_y_train[:train_val_split]
    self.X_val   = self.all_X_train[train_val_split:]
    self.y_val   = self.all_y_train[train_val_split:]

  def _shuffle_train(self):
    perm = np.arange(self.X_train.shape[0])
    np.random.shuffle(perm)
    self.X_train = self.X_train[perm]
    self.y_train = self.y_train[perm]

  def train_batches(self, batch_size):
    self._shuffle_train()
    for start in xrange(0, self.X_train.shape[0], batch_size):
      yield ( self.X_train[start: start + batch_size],
              self.y_train[start: start + batch_size])
