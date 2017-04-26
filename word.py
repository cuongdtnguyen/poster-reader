from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import gridspec
from classifiers import *
from predict import Predictor
from trainRecognizer import CHARACTER_CODES

try:
  import cPickle as pickle
except:
  import pickle

DETECTOR_MODEL_DIR = 'selectedModels/detectorModel'
RECOGNIZER_MODEL_DIR = 'selectedModels/recognizerModel'
DETECTOR_MODEL = DETECTOR_MODEL_DIR + '/detectorModel.ckpt-2048'
RECOGNIZER_MODEL = RECOGNIZER_MODEL_DIR + '/recognizerModel.ckpt-6510'
DETECTOR_NORM = DETECTOR_MODEL_DIR + '/normalization.pickle'
RECOGNIZER_NORM = RECOGNIZER_MODEL_DIR + '/normalization.pickle'
WINDOW_SIZE_RATIO = 0.75
NMS_WIDTH = 2
SLIDING_STEP = 1
IMAGE_SIZE = 32 * 32

def preprocess(X, norm_path):
  with open(norm_path, 'rb') as f:
    mean, std = pickle.load(f)

  X = X.reshape(IMAGE_SIZE)
  X = (X - mean) / std
  return X


def nms(signal, w):
  new_signal = np.zeros_like(signal)
  for i in range(len(signal)):
    l = max(0, i - w)
    r = min(len(signal), i + w + 1)
    mx = np.max(signal[l:r])
    new_signal[i] = max(signal[i], 0) if signal[i] >= mx else 0
  return new_signal


def slide_window(imgs, model, model_path, num_classes, normalization_path, as_prob):
  scores = []
  for img in imgs:
    H, W = img.shape
    windows = []
    step = SLIDING_STEP
    window_width = int(H*WINDOW_SIZE_RATIO)
    for i in range(0, W - window_width + 1, step):
      window = scipy.misc.imresize(img[:, i:i + window_width],
                                  (32, 32), interp='bilinear')
      window = preprocess(window, normalization_path)
      windows.append(window)

    if len(windows) > 0:
      predictor = Predictor(model, model_path, num_classes)
      scores.append(predictor.predict(windows, as_prob=as_prob))
      predictor.close()

  return scores


def word_score(word, reg_map):
  def to_char_index(c):
    return CHARACTER_CODES.index(ord(c))

  char_list = map(to_char_index, word)

  likelihood = reg_map[:, char_list]

  score = np.zeros_like(likelihood)
  score[:, 0] = likelihood[:, 0]
  for j in range(1, score.shape[1]):
    for i in range(j, score.shape[0]):
      score[i, j] = max(score[i - 1, j],
                        score[i - 1, j - 1] + likelihood[i, j]
                                            + likelihood[i-1,j])

  return score[-1, -1]


def segment(imgs):
  model = ConvNet((32, 32), 2)
  scores = slide_window(imgs, model,
                       model_path=DETECTOR_MODEL,
                       normalization_path=DETECTOR_NORM,
                       num_classes=2,
                       as_prob=True)

  return scores


def recognize(imgs, lexicon, show_graph_first_one=False, verbose=False):

  if verbose:
    print('Calculating detect scores...')
  detect_scores = segment(imgs)

  model = ConvNet3((32, 32), len(CHARACTER_CODES))
  if verbose:
    print('Calculating recognize scores...')
  recognize_scores = slide_window(imgs, model,
                                 model_path=RECOGNIZER_MODEL,
                                 normalization_path=RECOGNIZER_NORM,
                                 num_classes=len(CHARACTER_CODES),
                                 as_prob=True)

  def margin(v):
    """Returns the difference of the largest and second largest element"""
    return np.max(v) - np.partition(v.flatten(), -2)[-2]

  results = []
  for i, img in enumerate(imgs):

    conf_margin_coarse = map(margin, recognize_scores[i])
    conf_margin_refined = nms(conf_margin_coarse, NMS_WIDTH)

    detect_mask = detect_scores[i][:, 0]
    conf_margin = np.copy(conf_margin_refined)
    conf_margin[detect_mask <= 0.5] = 0

    reduced_score = recognize_scores[i][conf_margin > 0]

    if reduced_score.shape[0] == 0:
      results.append(("", -1))
      continue
    print(map(lambda x: chr(CHARACTER_CODES[x]), np.argmax(reduced_score, axis=1)))

    selected = None
    max_score = 0
    for word in lexicon:
      score = word_score(word, reduced_score)
      print(word, score)
      if score > max_score:
        max_score = score
        selected = word

    results.append((selected, max_score))

    if show_graph_first_one and i == 0:
      H, W = img.shape
      fig, ax = plt.subplots(5, 1,
                             figsize=(7, 8), dpi=90,
                             gridspec_kw={'hspace': 1.0})

      ax[0].imshow(img, cmap='gray', interpolation='nearest', aspect='auto')
      ax[0].set_xlabel('image of a word')
      ax[1].plot(conf_margin_coarse, 'red')
      ax[1].set_xlabel('coarse confidence margin')
      ax[2].plot(conf_margin_refined, 'red')
      ax[2].set_xlabel('refined confidence margin after NMS')
      ax[3].plot(detect_mask, 'green')
      ax[3].set_xlabel('detection score')
      ax[4].plot(conf_margin, 'blue')
      ax[4].set_xlabel('confidence margin after applying detection score')
      plt.show(block=False)

  return results


if __name__=='__main__':
  img_paths = ['examples/research.png',
               'examples/valuable.jpg',
               'examples/prototype.jpg',
               'examples/maintenance.jpg',
               'examples/graduate.png',
               'examples/good.jpg',
               'examples/celebrating.png',
               'examples/butler.jpg',
               'examples/bioengineering.jpg']

  imgs = []
  for p in img_paths:
    img = scipy.misc.imread(p, mode='L')
    H, W = img.shape
    img = scipy.misc.imresize(img, (32, int(32 * W / H)), interp='bilinear')
    imgs.append(img)

  lexicon = ['valuable', 'graduate', 'bioengineering', 'Research',
             'Reach', 'Researches', 'Reassure', 'research', 'Good', 'butler',
             'BUTLER', 'Butler', 'Celebrating', 'celebrating', 'maintenance',
             'prototype', 'PROTOTYPE']

  results = recognize(imgs, lexicon, show_graph_first_one=False, verbose=True)
  print(results)
  plt.show()
