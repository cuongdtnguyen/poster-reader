class Model(object):
  """An interface for a classifying model

  This interface was inspired from this TensorFlow tutorial:
  https://www.tensorflow.org/get_started/mnist/mechanics

  inference:  takes in an input_data op that produces the training data,
              for example: tf.placeholder, and produces a logits op that
              calculates the score of the given data by passing it through the
              model.

  loss:       takes in a logits op and a labels op. Labels op produces a label
              for its corresponding instance in the training data. this method
              combines these two ops to produce a loss calculating op.

  training:   takes in the loss op and applies an optimizer on the loss.

  evaluation: takes in logits and labels ops and produces an op that report
              accuracy of the current model
  """

  def inference(self, input_data, **kwargs):
    raise NotImplementedError

  def loss(self, logits, labels, **kwargs):
    raise NotImplementedError

  def training(self, loss, learning_rate, **kwargs):
    raise NotImplementedError

  def evaluation(self, logits, labels, **kwargs):
    raise NotImplementedError