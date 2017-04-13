from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from data_utils import *
from classifiers import *
from trainer import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epoch')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_integer('val_batch_size', 500, 'Batch size for validation')
tf.app.flags.DEFINE_boolean('verbose', True, 'Verbose')
tf.app.flags.DEFINE_float('lr', 0.05, 'Learning rate')
tf.app.flags.DEFINE_integer('log_freq', 5,
                            'Log frequency every number of epochs')
tf.app.flags.DEFINE_float('reg', 0.00, 'Regularization')
tf.app.flags.DEFINE_string('save_path', 'recognizerModel/recognizerModel.ckpt',
                           'Path to file that this model will be saved to')

# CHARACTER_CODES = range(ord('0'),ord('9')+1) + range(ord('A'),ord('Z')+1) + range(ord('a'),ord('z')+1)
CHARACTER_CODES = range(ord('0'),ord('9')+1) + range(ord('a'),ord('z')+1)

def uncapitalize(a):
  return map(lambda c : ord(chr(c).lower()), a)

def convert_to_index(a):
  return map(lambda c : CHARACTER_CODES.index(c), a)

def convert_to_code(a):
  return map(lambda i : CHARACTER_CODES[i], a)

def main(argv=None):
  data = ImageDataset('dataset/recognizerData')

  # Save normalization values so that predictor can use later
  data.save_normalize(os.path.join(os.path.dirname(FLAGS.save_path),
                                 'normalization.pickle'))

  data.y_train = convert_to_index(uncapitalize(data.y_train))
  data.y_val = convert_to_index(uncapitalize(data.y_val))
  data.y_test = convert_to_index(uncapitalize(data.y_test))
  num_label = len(CHARACTER_CODES)
  data.y_train = to_one_hot(data.y_train, num_label)
  data.y_val = to_one_hot(data.y_val, num_label)
  data.y_test = to_one_hot(data.y_test, num_label)

  # model = FullyConnectedNet(data.X_train.shape[1],
  #                           [100, 100, 100],
  #                           data.y_train.shape[1],
  #                           reg=FLAGS.reg)

  #model = ConvNet((32, 32), len(CHARACTER_CODES))
  model = ConvNet2((32, 32), len(CHARACTER_CODES))
  train(model, data,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size,
    val_batch_size=FLAGS.val_batch_size,
    learning_rate=FLAGS.lr,
    log_freq=FLAGS.log_freq,
    verbose=FLAGS.verbose,
    save_path=FLAGS.save_path)

if __name__=='__main__':
  tf.app.run()