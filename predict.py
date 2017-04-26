from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
  import cPickle as pickle
except:
  import pickle

from sklearn.metrics import confusion_matrix
from trainRecognizer import preprocess_data as preprocess_data_recognizer
from trainDetector import preprocess_data as preprocess_data_detector
from trainRecognizer import CHARACTER_CODES
from classifiers import *
from data_utils import *

DETECTOR_MODEL_DIR = 'selectedModels/detectorModel'
RECOGNIZER_MODEL_DIR = 'selectedModels/recognizerModel'
# DETECTOR_MODEL = DETECTOR_MODEL_DIR + '/detectorModel.ckpt-2048'
DETECTOR_MODEL = DETECTOR_MODEL_DIR + '/detectorModel.ckpt-1800'
#RECOGNIZER_MODEL = RECOGNIZER_MODEL_DIR + '/recognizerModel.ckpt-6510'
RECOGNIZER_MODEL = RECOGNIZER_MODEL_DIR + '/recognizerModel.ckpt-3840'
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_text=True):
  plt.figure(figsize=(12, 9), dpi=70)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  if show_text:
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


def evaluateDetector():
  with open(DETECTOR_NORM, 'rb') as f:
    mean, std = pickle.load(f)

  data = ImageDataset('dataset/detectorData', test_only=True,
                      mean_train=mean, std_train=std)
  data = preprocess_data_detector(data)
  model = ConvNet2((32, 32), 2)
  predictor = Predictor(model, DETECTOR_MODEL, 2)
  predicted = predictor.predict(data.X_test)
  accuracy = predictor.evaluate(data.X_test, data.y_test)
  predictor.close()

  predicted = np.argmax(predicted, axis=1)
  ground_truth = np.argmax(data.y_test, axis=1)
  cnf_matrix = confusion_matrix(ground_truth, predicted)
  plot_confusion_matrix(cnf_matrix, classes=['Non-Text', 'Text'],
                        title='Confusion matrix for detector')

  return accuracy


def evaluateRecognizer():
  with open(RECOGNIZER_NORM, 'rb') as f:
    mean, std = pickle.load(f)

  data = ImageDataset('dataset/recognizerData', test_only=True,
                      mean_train=mean, std_train=std)
  data = preprocess_data_recognizer(data)
  model = ConvNet3((32, 32), len(CHARACTER_CODES))
  predictor = Predictor(model, RECOGNIZER_MODEL, len(CHARACTER_CODES))
  predicted = predictor.predict(data.X_test, as_prob=True)
  accuracy = predictor.evaluate(data.X_test, data.y_test)
  predictor.close()

  predicted = np.argmax(predicted, axis=1)
  ground_truth = np.argmax(data.y_test, axis=1)
  cnf_matrix = confusion_matrix(ground_truth, predicted)
  classes = map(lambda c: chr(c), CHARACTER_CODES)
  plot_confusion_matrix(cnf_matrix, classes=classes, cmap=plt.cm.jet,
                        title='Confusion matrix for recognizer', show_text=False)


  return accuracy


def main():
  detectorAcc = evaluateDetector()
  recognizerAcc = evaluateRecognizer()
  print('Accuracy on test set of the detector:', detectorAcc)
  print('Accuracy on test set of the recognizer:', recognizerAcc)

  plt.show()


if __name__=='__main__':
  main()
