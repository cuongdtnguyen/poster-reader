from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

try:
  import cPickle as pickle
except:
  import pickle

import pandas as pd
import numpy as np
import scipy.misc

class Data:

  # TODO: preprocess data after loading
  def load_from_image(self, prefix):
    train_label_path  = os.path.join(prefix, 'trainLabels.csv')
    test_label_path   = os.path.join(prefix, 'testLabels.csv')

    self.all_y_train = pd.read_csv(train_label_path, header=None, index_col=0)
    self.y_test  = pd.read_csv(test_label_path, header=None, index_col=0)

    N_all_train = self.all_y_train.size
    N_test  = self.y_test.size

    print('Loading train set...')
    all_X_train = []
    for i in xrange(1, N_all_train + 1):
      img_path = os.path.join(prefix, 'train', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      all_X_train.append(I.reshape((1, -1)))
    print('Finished loading train set')

    self.all_X_train = np.array(all_X_train)

    self._split_train_val()

    print('Loading test set...')
    X_test = []
    for i in xrange(1, N_test + 1):
      img_path = os.path.join(prefix, 'test', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      X_test.append(I.reshape((1, -1)))
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
    train_val_split = int(self.all_y_train.size * 0.8)
    self.X_train = self.all_X_train[:train_val_split]
    self.X_val   = self.all_X_train[train_val_split:]
    self.y_train = self.all_y_train[:train_val_split]
    self.y_val   = self.all_y_train[train_val_split:]

  def train_batches(self, batch_size):
    for start in xrange(0, self.X_train.size, batch_size):
      yield ( self.X_train[start: start + batch_size],
              self.y_train[start: start + batch_size])


