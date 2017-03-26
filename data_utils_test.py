from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils import Data

data = Data()
data.load_images('dataset/detectorData')

print(data.X_train.shape)
print(data.X_train[:10])
print(data.y_train.shape)
print(data.y_train[:10])