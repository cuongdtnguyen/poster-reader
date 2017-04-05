class convolu_net(object):

  def conv2d(self, input_data, weight, bias, stride):
    raise NotImplementedError

  def maxmaxpool2d(self, input_data, k):
    raise NotImplementedError

  def conv_net(self, input_data, weight, bias):
    raise NotImplementedError