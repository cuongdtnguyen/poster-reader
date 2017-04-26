from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from trainRecognizer import preprocess_data as preprocess_data_recognizer
from trainDetector import preprocess_data as preprocess_data_detector
from trainRecognizer import CHARACTER_CODES

try:
  import cPickle as pickle
except:
  import pickle

from classifiers import *
from data_utils import *

DETECTOR_MODEL_DIR = 'selectedModels/detectorModel'
RECOGNIZER_MODEL_DIR = 'selectedModels/recognizerModel'
DETECTOR_MODEL = DETECTOR_MODEL_DIR + '/detectorModel.ckpt-2048'
RECOGNIZER_MODEL = RECOGNIZER_MODEL_DIR + '/recognizerModel.ckpt-6510'
DETECTOR_NORM = DETECTOR_MODEL_DIR + '/normalization.pickle'
RECOGNIZER_NORM = RECOGNIZER_MODEL_DIR + '/normalization.pickle'

class Predictor:

  def __init__(self, model, ckpt_path, num_class):
    tf.reset_default_graph()
    self.sess = tf.Session()
    self.X_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE**2))
    self.y_placeholder = tf.placeholder(tf.int32, shape=(None, num_class))
    self.keep_prob     = tf.placeholder(tf.float32)
    self.logits, _     = model.inference(self.X_placeholder, keep_prob=self.keep_prob)
    self.softmax       = tf.nn.softmax(self.logits)
    self.eval_correct  = model.evaluation(self.logits, self.y_placeholder)

    saver = tf.train.Saver()
    saver.restore(self.sess, ckpt_path)

  def predict(self, X, as_prob=True):
    if as_prob:
      score = self.sess.run(self.softmax, feed_dict={ self.X_placeholder: X,
                                                      self.keep_prob: 1.0 })
    else:
      score = self.sess.run(self.logits, feed_dict={ self.X_placeholder: X,
                                                     self.keep_prob: 1.0 })
    return score

  def evaluate(self, X_test, y_test):
    test_acc = self.sess.run(self.eval_correct,
                             feed_dict={ self.X_placeholder: X_test,
                                         self.y_placeholder: y_test,
                                         self.keep_prob: 1.0})
    return test_acc

  def close(self):
    self.sess.close()


def evaluateDetector():
  with open(DETECTOR_NORM, 'rb') as f:
    mean, std = pickle.load(f)

  data = ImageDataset('dataset/detectorData', test_only=True,
                      mean_train=mean, std_train=std)
  data = preprocess_data_detector(data)
  model = ConvNet((32, 32), 2)
  predictor = Predictor(model, DETECTOR_MODEL, 2)
  accuracy = predictor.evaluate(data.X_test, data.y_test)
  predictor.close()

  return accuracy


def evaluateRecognizer():
  with open(RECOGNIZER_NORM, 'rb') as f:
    mean, std = pickle.load(f)

  data = ImageDataset('dataset/recognizerData', test_only=True,
                      mean_train=mean, std_train=std)
  data = preprocess_data_recognizer(data)
  model = ConvNet3((32, 32), len(CHARACTER_CODES))
  predictor = Predictor(model, RECOGNIZER_MODEL, len(CHARACTER_CODES))
  accuracy = predictor.evaluate(data.X_test, data.y_test)
  predictor.close()

  return accuracy


def main():
  detectorAcc = evaluateDetector()
  recognizerAcc = evaluateRecognizer()
  print('Accuracy on test set of the detector:', detectorAcc)
  print('Accuracy on test set of the recognizer:', recognizerAcc)


if __name__=='__main__':
  main()
