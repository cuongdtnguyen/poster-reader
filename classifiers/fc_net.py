import tensorflow as tf

class FullyConnectedNet(Model):

  def __init__(self, input_dims, hidden_dims, num_classes,
               reg=0.0, weight_scale=1e-3, dtype=tf.float32):
    self.input_dims = input_dims
    self.hidden_dims = hidden_dims
    self.num_classes = num_classes
    self.reg = reg
    self.weight_scale = weight_scale
    self.dtype = dtype

  def inference(input_data):
    num_layers = len(self.hidden_dims) + 1

    # Fisrt hidden layer
    with tf.name_scope('hidden1'):
      weights = tf.Variable(
        tf.truncated_normal([self.input_dims, hidden_dims[0]],
                            stddev=self.weight_scale), # Use 1 / sqrt(input_dims)?
        name='weights')
      bias = tf.Variable(tf.zeros([hidden_dims[0]]),
                         name='biases')
      hidden1 = tf.nn.relu(tf.matmul(input_data, weights) + biases)

    # Hidden layer from 2 -> number of layers - 1
    prev_hidden = hidden1
    for l in range(2, num_layers - 1):

      with tf.name_scope('hidden' + str(l)):
        weights = tf.Variable(
        tf.truncated_normal([hidden_dims[l - 2], hidden_dims[l - 1]],
                            stddev=self.weight_scale), # Use 1 / sqrt(hidden_dims[l - 2]?
        name='weights')
        bias = tf.Variable(tf.zeros([hidden_dims[l - 1]]),
                           name='biases')
        hidden = tf.nn.relu(tf.matmul(prev_hidden, weights) + biases)
        prev_hidden = hidden

    # Softmax layer
    with tf.name_scope('softmax_linear'):
      weights = tf.Variable(
        tf.truncated_normal([self.input_dims, hidden_dims[0]],
                            stddev=self.weight_scale), # Use 1 / sqrt(input_dims)?
        name='weights')
      bias = tf.Variable(tf.zeros([hidden_dims[0]]),
                         name='biases')
      logits = tf.matmul(prev_hidden, weights) + biases
    return logits

  def loss(logits, labels):
    # required for sparse_softmax_cross_entropy_with_logits
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

  def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

  def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))



