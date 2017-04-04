import tensorflow as tf

from model import Model

class FullyConnectedNet(Model):

  def __init__(self, input_dims, hidden_dims, num_classes,
               reg=0.00, weight_scale=1e-3):
    self.input_dims = input_dims
    self.hidden_dims = hidden_dims
    self.num_classes = num_classes
    self.reg = reg
    self.weight_scale = weight_scale

  def inference(self, input_data):
    num_layers = len(self.hidden_dims) + 1

    # Fisrt hidden layer
    with tf.name_scope('hidden1'):
      weights = tf.Variable(
        tf.truncated_normal([self.input_dims, self.hidden_dims[0]],
                            stddev=self.weight_scale),
        name='weights')
      biases = tf.Variable(tf.zeros([self.hidden_dims[0]]),
                         name='biases')
      hidden1 = tf.nn.relu(tf.matmul(input_data, weights) + biases)
      l2_reg  = tf.nn.l2_loss(weights)

    # Hidden layer from 2 -> number of layers - 1
    prev_hidden = hidden1
    for l in range(2, num_layers):

      with tf.name_scope('hidden' + str(l)):
        weights = tf.Variable(
        tf.truncated_normal([self.hidden_dims[l - 2], self.hidden_dims[l - 1]],
                            stddev=self.weight_scale),
        name='weights')
        biases = tf.Variable(tf.zeros([self.hidden_dims[l - 1]]),
                           name='biases')
        hidden = tf.nn.relu(tf.matmul(prev_hidden, weights) + biases)

        l2_reg  = l2_reg + tf.nn.l2_loss(weights)
        prev_hidden = hidden

    # Softmax layer
    with tf.name_scope('softmax_linear'):
      weights = tf.Variable(
        tf.truncated_normal([self.hidden_dims[-1], self.num_classes],
                            stddev=self.weight_scale),
        name='weights')
      biases = tf.Variable(tf.zeros([self.num_classes]), name='biases')
      logits = tf.matmul(prev_hidden, weights) + biases

      l2_reg  = l2_reg + tf.nn.l2_loss(weights)

    return logits, l2_reg

  def loss(self, logits, labels, l2_reg=None):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')

    if l2_reg is not None:
      return tf.reduce_mean(cross_entropy + self.reg * l2_reg,
                            name='xentropy_mean_reg')
    else:
      return tf.reduce_mean(cross_entropy, name='xentropy_mean')

  def training(self, loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

  def evaluation(self, logits, labels):
    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32), name='evaluation')




