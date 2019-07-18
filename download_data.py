import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import warnings

from IPython.display import Image
from sklearn.cluster import MeanShift

import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import cv2
import os.path
import warnings
from distutils.version import LooseVersion
import glob
from PIL import Image

import cv2

data_dir = "/content/cityscapes/data/"

def preprocessinglabel(path):
  frame = cv2.imread(path)
  if frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  cv2.imwrite(path,frame)

if os.path.isdir( data_dir + "gtFine/") and  os.path.isdir(data_dir + "leftImg8bit/"):
    pass
else:
  ! wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=anhdhbn&password=Honganh99&submit=Login' https://www.cityscapes-dataset.com/login/
  ! wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P data
  ! unzip -qq -o data/gtFine_trainvaltest.zip -d {data_dir}
  ! wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P data
  ! unzip -qq -o data/leftImg8bit_trainvaltest.zip -d {data_dir}
  
  for path in Y_train:
    preprocessinglabel(path)
  for path in Y_val:
    preprocessinglabel(path)
  for path in Y_test:
    preprocessinglabel(path)
