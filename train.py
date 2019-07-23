import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
import time

start_time = time.time()
project_dir = "/content/cityscapes/segmentation/"

data_dir = project_dir + "data/"

preprocess_data_dir = "./preprocess-data/"

batch_size = 4

img_height = 512
img_width = 1024

no_of_classes = 20

epochs = 5


# train_mean_channels = pickle.load(open(project_dir + "data/mean_channels.pkl"))
train_mean_channels = pickle.load(open(preprocess_data_dir + "mean_channels.pkl", "rb"))

# load the training data from disk:
# train_img_paths = pickle.load(open(data_dir + "train_img_paths.pkl"))
# train_trainId_label_paths = pickle.load(open(data_dir + "train_trainId_label_paths.pkl"))

train_img_paths = pickle.load(open(preprocess_data_dir + "train_img_paths.pkl", "rb"))
train_trainId_label_paths = pickle.load(open(preprocess_data_dir + "train_trainId_label_paths.pkl", "rb"))
train_data = list(zip(train_img_paths, train_trainId_label_paths))


# compute the number of batches needed to iterate through the training data:
no_of_train_imgs = len(train_img_paths)
no_of_batches = int(no_of_train_imgs/batch_size)

# load the validation data from disk:
# val_img_paths = pickle.load(open(data_dir + "val_img_paths.pkl"))
# val_trainId_label_paths = pickle.load(open(data_dir + "val_trainId_label_paths.pkl"))

val_img_paths = pickle.load(open(preprocess_data_dir + "val_img_paths.pkl", "rb"))
val_trainId_label_paths = pickle.load(open(preprocess_data_dir + "val_trainId_label_paths.pkl", "rb"))

val_data = list(zip(val_img_paths, val_trainId_label_paths))

# compute the number of batches needed to iterate through the val data:
no_of_val_imgs = len(val_img_paths)
no_of_val_batches = int(no_of_val_imgs/batch_size)

# define params needed for label to onehot label conversion:
layer_idx = np.arange(img_height).reshape(img_height, 1)
component_idx = np.tile(np.arange(img_width), (img_height, 1))


def train_data_iterator():
    idx = range(len(train_data))
    random.shuffle(train_data)
    train_img_paths, train_trainId_label_paths = zip(*train_data)
    while True:
        path_idx = np.random.choice(a = len(train_data), size = batch_size)
        # get and yield the next batch_size imgs and onehot labels from the train data:
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)

        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(train_img_paths[path_idx[i]], -1)
#             print(train_img_paths[path_idx[i]], train_trainId_label_paths[path_idx[i]])
            img = img - train_mean_channels
            batch_imgs[i] = img

            # read the next label:
            trainId_label = cv2.imread(train_trainId_label_paths[path_idx[i]], -1)

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, trainId_label] = 1
            batch_onehot_labels[i] = onehot_label
        path_idx = np.random.choice(a = len(train_data), size = batch_size)

        yield (batch_imgs, batch_onehot_labels)

from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import model as m

input_img = Input((img_height, img_width, 3), name='img')
model = m.get_unet(input_img, n_filters=32, dropout=0.1, batchnorm=True)

# model.compile(optimizer=Adam(), loss="binary_crossentropy")
# model.compile(optimizer=Adam(), loss=[mean_iou])
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

his = model.fit_generator(generator=train_data_iterator(), epochs=epochs, steps_per_epoch=no_of_batches)

print("--- %s seconds ---" % (time.time() - start_time))
