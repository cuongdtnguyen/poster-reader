import os
import pickle

import pandas as pd
import numpy as np
import scipy.misc

class Data:

  def load_from_image(self, prefix):
    train_label_path  = os.path.join(prefix, 'trainLabels.csv')
    test_label_path   = os.path.join(prefix, 'testLabels.csv')

    self.y_train = pd.read_csv(train_label_path, header=None, index_col=0)
    self.y_test  = pd.read_csv(test_label_path, header=None, index_col=0)

    self.N_train = self.y_train.size
    self.N_test  = self.y_test.size

    print('Loading train set...')
    X_train = []
    for i in range(1, self.N_train + 1):
      img_path = os.path.join(prefix, 'train', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      X_train.append(I.reshape((1, -1)))

    self.X_train = np.array(X_train)
    print('Finished loading train set')

    print('Loading test set...')
    X_test = []
    for i in range(1, self.N_test + 1):
      img_path = os.path.join(prefix, 'test', str(i) + '.png')
      I = scipy.misc.imread(img_path, flatten=True)
      X_test.append(I.reshape((1, -1)))

    self.X_test = np.array(X_test)
    print('Finished loading test set')

  def pickle(self, prefix):
    with open(os.path.join(prefix, 'X_train.pickle'), 'wb') as f:
      pickle.dump(self.X_train, f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'wb') as f:
      pickle.dump(self.y_train, f)
    with open(os.path.join(prefix, 'X_test.pickle'), 'wb') as f:
      pickle.dump(self.X_test, f)
    with open(os.path.join(prefix, 'y_test.pickle'), 'wb') as f:
      pickle.dump(self.y_test, f)

  def unpickle(self, prefix):
    with open(os.path.join(prefix, 'X_train.pickle'), 'rb') as f:
      self.X_train = pickle.load(f)
    with open(os.path.join(prefix, 'y_train.pickle'), 'rb') as f:
      self.y_train = pickle.load(f)
    with open(os.path.join(prefix, 'X_test.pickle'), 'rb') as f:
      self.X_test = pickle.load(f)
    with open(os.path.join(prefix, 'y_test.pickle'), 'rb') as f:
      self.y_test = pickle.load(f)




