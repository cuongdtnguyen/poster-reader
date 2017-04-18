from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from data_utils import ImageDataset
from classifiers import *

VAL_THRES = 1e-4

def train(model, data, num_epochs, batch_size, val_batch_size, learning_rate,
          log_freq, verbose, save_path):
  """Trains a model with given data
  Args:
    model:          a TensorFlow op that represents a model
    data:           an ImageDataset object that contain data to be train
    batch_size:     size of a batch to be processed in a learning step
    learning_rate:  learning rate of gradietn descent or the like
    log_freq:       how many steps the log is printed
    verbose:        print information onto the screen or not
    save_path:      path to save the model and summary

  """
  num_dimension = data.X_train.shape[1]
  num_class     = data.y_train.shape[1]

  # Prepare placeholders for data
  X_placeholder = tf.placeholder(tf.float32, shape=(None, num_dimension))
  y_placeholder = tf.placeholder(tf.int32, shape=(None, num_class))
  keep_prob     = tf.placeholder(tf.float32)

  # Build up computational graph
  # logits, l2_reg = model.inference(X_placeholder)
  logits = model.inference(X_placeholder, keep_prob)
  # loss_op  = model.loss(logits, y_placeholder, l2_reg)
  loss_op = model.loss(logits, y_placeholder)
  train_op = model.training(loss_op, learning_rate)
  eval_correct = model.evaluation(logits, y_placeholder)


  init = tf.global_variables_initializer()

  saver = tf.train.Saver()

  with tf.Session() as sess:

    sess.run(init)
    step = 0
    old_val_acc = 0
    if verbose:
      print('Start training...')
    for epoch in range(num_epochs):

      for X_batch, y_batch in data.train_batches(batch_size):
        step += 1
        _, loss = sess.run([train_op, loss_op],
                           feed_dict={ X_placeholder: X_batch,
                                       y_placeholder: y_batch,
                                       keep_prob: 0.5 })

        train_loss = sess.run(loss_op,
                              feed_dict={ X_placeholder: X_batch,
                                          y_placeholder: y_batch,
                                          keep_prob: 1.0})
        train_acc = sess.run(eval_correct,
                             feed_dict={ X_placeholder: X_batch,
                                         y_placeholder: y_batch,
                                         keep_prob: 1.0 })

        if verbose and step % log_freq == 0:
          print('Step %d. Train loss: %f. Train acc: %f' % (step, train_loss, train_acc))


      sum_val_acc = 0
      num_val_batches = 0
      for X_batch, y_batch in data.val_batches(val_batch_size):
        num_val_batches += 1
        sum_val_acc += sess.run(eval_correct,
                                feed_dict={ X_placeholder: X_batch,
                                            y_placeholder: y_batch,
                                            keep_prob: 1.0 })
      val_acc = sum_val_acc / num_val_batches
      print('Epoch %d/%d. Validation accuracy: %f' % (
        epoch + 1, num_epochs, val_acc))

      if abs(val_acc - old_val_acc) / val_acc < VAL_THRES:
        learning_rate = 0.1 * learning_rate
        train_op = model.training(loss_op, learning_rate)
        print('New learning rate:', learning_rate)


      # Save the model
      if not os.path.exists(os.path.dirname(save_path)):
          os.makedirs(os.path.dirname(save_path))
      saved = saver.save(sess, save_path, global_step=step)

    print('Model saved in file: %s' % saved)


