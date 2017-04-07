from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from trainRecognizer import CHARACTER_CODES

try:
  import cPickle as pickle
except:
  import pickle

from classifiers import *
from data_utils import *

IMAGE_SIZE = 32 * 32

class Predictor:

  def __init__(self, model, ckpt_path):
    print("Creating predictor")
    tf.reset_default_graph()
    self.sess = tf.Session()
    self.X_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE))
    self.keep_prob = tf.placeholder(tf.float32)
    # self.logits, _ = model.inference(self.X_placeholder)
    self.logits = model.inference(self.X_placeholder, self.keep_prob)
    self.softmax = tf.nn.softmax(self.logits)

    saver = tf.train.Saver()
    saver.restore(self.sess, ckpt_path)

  def predict(self, X):
    score = self.sess.run(self.softmax, feed_dict={ self.X_placeholder: X,
                                                    self.keep_prob: 1.0 })
    return score

  def close(self):
    self.sess.close()


def predictDetector(X):
  with open('detectorModel/normalization.pickle', 'rb') as f:
    mean, std = pickle.load(f)

  X = X.reshape((-1, IMAGE_SIZE))
  X = (X - mean) / std

  # model = FullyConnectedNet(IMAGE_SIZE, [100, 100], 2)
  model = ConvNet((32, 32), 2)
  predictor = Predictor(model, 'detectorModel/detectorModel.ckpt-2125')
  score = predictor.predict(X)
  predictor.close()

  return score


def predictRecognizer(X):
  with open('recognizerModel/normalization.pickle', 'rb') as f:
    mean, std = pickle.load(f)

  X = X.reshape((-1, IMAGE_SIZE))
  X = (X - mean) / std

  # model = FullyConnectedNet(IMAGE_SIZE, [100, 100], len(CHARACTER_CODES))
  model = ConvNet((32, 32), len(CHARACTER_CODES))
  predictor = Predictor(model, 'recognizerModel/recognizerModel.ckpt-5100')
  score = predictor.predict(X)
  predictor.close()

  return score

def main():
  for i in range(1, 11):
    I = load_image('examples/c%d.png' % (i), preprocess=True).reshape((1, -1))
    # print(predictDetector(I))
    score = predictRecognizer(I)
    idx = np.argsort(score, axis=1)[0, -1:-6:-1]
    for t in idx:
      print('%s: %f'%(chr(CHARACTER_CODES[t]), score[0, t]))
    print()
  # I = load_image('examples/E.png', preprocess=True)
  # score = predictRecognizer(I)
  # idx = np.argsort(score, axis=1)[0, -1:-6:-1]
  # for t in idx:
  #   print('%s: %f'%(chr(CHARACTER_CODES[t]), score[0, t]))


if __name__=='__main__':
  main()
