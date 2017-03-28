from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_utils import ImageDataset
from classifiers import *

# TODO: infer this from data instead
IMAGE_SIZE = 32 * 32

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epoch')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_boolean('verbose', True, 'Verbose')
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate')
tf.app.flags.DEFINE_integer('log_freq', 5, 'Log frequency every number of epochs')

def train(model, data):

  X_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                    IMAGE_SIZE))
  y_placeholder = tf.placeholder(tf.int32, shape=(None,))

  logits, l2_reg = model.inference(X_placeholder)
  loss_op  = model.loss(logits, y_placeholder, l2_reg)
  train_op = model.training(loss_op, FLAGS.lr)
  eval_correct = model.evaluation(logits, y_placeholder)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(FLAGS.num_epochs):
      for X_batch, y_batch in data.train_batches(FLAGS.batch_size):
        _, loss = sess.run([train_op, loss_op], feed_dict={ X_placeholder: X_batch,
                                                            y_placeholder: y_batch })

      if epoch % FLAGS.log_freq == 0:
        train_loss = sess.run(loss_op, feed_dict={ X_placeholder: data.X_train,
                                                   y_placeholder: data.y_train})
        train_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_train,
                                                       y_placeholder: data.y_train })
        val_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_val,
                                                     y_placeholder: data.y_val })

        print('(Epoch %d/%d). Train loss: %f. Train acc: %f; Val acc: %f' % (
          epoch, FLAGS.num_epochs, train_loss, train_acc / float(data.X_train.shape[0]),
                                               val_acc / float(data.X_val.shape[0]) ))

def main(argv=None):
  model = FullyConnectedNet(IMAGE_SIZE, [100, 100, 100], 2)
  data = ImageDataset('dataset/detectorData')
  train(model, data)

if __name__=='__main__':
  tf.app.run()