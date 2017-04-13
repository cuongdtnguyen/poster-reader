from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .model import Model

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='SAME')

class ConvNet2(Model):

  def __init__(self, image_size, num_classes):
    self.image_size = image_size
    self.num_classes = num_classes

  def inference(self, input_data, keep_prob):
    H, W = self.image_size

    x_image = tf.reshape(input_data, [-1, H, W, 1])

    # N x 32 x 32 x 32
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # N x 32 x 32 x 32
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # N x 16 x 16 x 32
    h_pool1 = max_pool_2x2(h_conv2)

    # N x 16 x 16 x 64
    W_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # N x 16 x 16 x 64
    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # N x 8 x 8 x 64
    h_pool2 = max_pool_2x2(h_conv4)

    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, self.num_classes])
    b_fc2 = weight_variable([self.num_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv

  def loss(self, logits, labels):
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

  def training(self, loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    return train_step

  def evaluation(self, logits, labels):
    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
