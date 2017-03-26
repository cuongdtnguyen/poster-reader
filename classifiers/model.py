class Model(object):

  def inference(self, input_data):
    raise NotImplementedError

  def loss(self, logits, labels):
    raise NotImplementedError

  def training(self, loss, learning_rate):
    raise NotImplementedError

  def evaluation(self, logits, labels):
    raise NotImplementedError