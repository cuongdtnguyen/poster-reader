from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import convolu_net
from data_utils import ImageDataset
from classifiers import *



def train(model, data, num_epochs, batch_size, learning_rate, log_freq, verbose):

  num_dimension = data.X_train.shape[1]
  num_class     = data.y_train.shape[1]
  dropout = 0.75  # Dropout, probability to keep units

  X_placeholder = tf.placeholder(tf.float32, shape=(None, num_dimension))
  y_placeholder = tf.placeholder(tf.int32, shape=(None, num_class))
  keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

  # Store layers weight & bias
  weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 4096 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4096, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_class]))
  }

  biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_class]))
  }
  # logits, l2_reg = model.inference(X_placeholder)
  logits_, l2_reg = convolu_net.conv_net(X_placeholder, weights, biases, keep_prob)
  loss_op  = model.loss(logits_, y_placeholder, l2_reg)
  train_op = model.training(loss_op, learning_rate)
  eval_correct = model.evaluation(logits_, y_placeholder)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
      for X_batch, y_batch in data.train_batches(batch_size):
        _, loss = sess.run([train_op, loss_op], feed_dict={ X_placeholder: X_batch,
                                                            y_placeholder: y_batch,
                                                            keep_prob: dropout})
        # Run optimization op (backprop)
        # sess.run(optimizer, feed_dict={ X_placeholder: X_batch,
        #                                 y_placeholder: y_batch })

      if epoch % log_freq == 0:
        train_loss = sess.run(loss_op, feed_dict={ X_placeholder: data.X_train,
                                                   y_placeholder: data.y_train,
                                                   keep_prob: 1. })
        train_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_train,
                                                       y_placeholder: data.y_train,
                                                       keep_prob: 1.})
        val_acc = sess.run(eval_correct, feed_dict={ X_placeholder: data.X_val,
                                                     y_placeholder: data.y_val,
                                                     keep_prob: 1. })
        # Calculate batch loss and accuracy
        # loss, acc = sess.run([cost, accuracy], feed_dict={ X_placeholder: X_batch,
        #                                                   y_placeholder: y_batch })

        print('(Epoch %d/%d). Train loss: %f. Train acc: %f; Val acc: %f' % (
          epoch, num_epochs, train_loss, train_acc, val_acc))