from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from data_utils import *
from classifiers import *
from trainer import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epoch')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_boolean('verbose', True, 'Verbose')
tf.app.flags.DEFINE_float('lr', 0.05, 'Learning rate')
tf.app.flags.DEFINE_integer('log_freq', 5, 'Log frequency every number of epochs')
tf.app.flags.DEFINE_float('reg', 0.00, 'Regularization')
tf.app.flags.DEFINE_string('save_path', 'detectorModel/detectorModel.ckpt', 'Path to file that this model will be saved to')

def main(argv=None):
  data = ImageDataset('dataset/detectorData')

  # Save normalization values so that predictor can use later
  data.save_normalize(os.path.join(os.path.dirname(FLAGS.save_path),
                                   'normalization.pickle'))

  # Zero-based index for labels
  data.y_train -= 1
  data.y_val   -= 1
  data.y_test  -= 1

  # Encode as one-hot vectors
  num_label = np.max(data.all_y_train) + 1
  data.y_train = to_one_hot(data.y_train, num_label)
  data.y_val = to_one_hot(data.y_val, num_label)
  data.y_test = to_one_hot(data.y_test, num_label)

  model = FullyConnectedNet(data.X_train.shape[1],
                            [100, 100],
                            data.y_train.shape[1],
                            reg=FLAGS.reg)

  train(model, data,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.lr,
    log_freq=FLAGS.log_freq,
    verbose=FLAGS.verbose,
    save_path=FLAGS.save_path)

if __name__=='__main__':
  tf.app.run()