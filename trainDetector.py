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

def preprocess_data(data):
  # Zero-based index for labels
  data.y_test  -= 1
  num_label = 2

  if not data.test_only:
    # Zero-based index for labels
    data.y_train -= 1
    data.y_val   -= 1
    # Encode as one-hot vectors

    data.y_train = to_one_hot(data.y_train, num_label)
    data.y_val = to_one_hot(data.y_val, num_label)

  data.y_test = to_one_hot(data.y_test, num_label)
  return data

def main(argv=None):
  data = ImageDataset('dataset/detectorData', train_val_ratio=FLAGS.split)

  # Save normalization values so that predictor can use later
  data.save_normalize(os.path.join(os.path.dirname(FLAGS.save_path),
                                   'normalization.pickle'))

  data = preprocess_data(data)

  models = [FullyConnectedNet(data.X_train.shape[1], [100, 100],
                              data.y_train.shape[1], reg=FLAGS.reg),
            ConvNet((32, 32), 2),
            ConvNet2((32, 32), 2),
            ConvNet3((32, 32), 2, reg=FLAGS.reg)]

  model = models[FLAGS.model]

  train(model, data,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size,
    val_batch_size=FLAGS.val_batch_size,
    learning_rate=FLAGS.lr,
    log_freq=FLAGS.log_freq,
    verbose=FLAGS.verbose,
    save_path=FLAGS.save_path)

if __name__=='__main__':
  tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epoch')
  tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
  tf.app.flags.DEFINE_integer('val_batch_size', 500, 'Batch size for validation')
  tf.app.flags.DEFINE_boolean('verbose', True, 'Verbose')
  tf.app.flags.DEFINE_float('lr', 0.05, 'Learning rate')
  tf.app.flags.DEFINE_integer('log_freq', 5, 'Log frequency every number of epochs')
  tf.app.flags.DEFINE_float('reg', 0.0, 'Regularization')
  tf.app.flags.DEFINE_string('save_path', 'detectorModel/detectorModel.ckpt',
                             'Path to file that this model will be saved to')
  tf.app.flags.DEFINE_integer('model', 1, 'Model to train with (0: fully-connected, 1: convnet1, 2: convnet2, 3: convnet3)')
  tf.app.flags.DEFINE_float('split', 0.85, 'Train/Val split ratio')
  tf.app.run()