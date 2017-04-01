from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_utils import ImageDataset
from classifiers import *


def train(model, data, num_epochs, batch_size, learning_rate, log_freq, verbose):

  num_dimension = data.X_train.shape[1]
  num_class     = data.y_train.shape[1]

  X_placeholder = tf.placeholder(tf.float32, shape=(None, num_dimension))
  y_placeholder = tf.placeholder(tf.int32, shape=(None, num_class))

  logits, l2_reg = model.inference(X_placeholder)
  loss_op  = model.loss(logits, y_placeholder, l2_reg)
  train_op = model.training(loss_op, learning_rate)
  eval_correct = model.evaluation(logits, y_placeholder)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
      for X_batch, y_batch in data.train_batches(batch_size):
        _, loss = sess.run([train_op, loss_op], feed_dict={ X_placeholder: X_batch,
                                                            y_placeholder: y_batch })

      if epoch % log_freq == 0:
        train_loss = sess.run(loss_op, feed_dict={ X_placeholder: data.X_train,
                                                   y_placeholder: data.y_train})
        train_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_train,
                                                       y_placeholder: data.y_train })
        val_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_val,
                                                     y_placeholder: data.y_val })

        print('(Epoch %d/%d). Train loss: %f. Train acc: %f; Val acc: %f' % (
          epoch, num_epochs, train_loss, train_acc, val_acc))