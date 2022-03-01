import cv2
import os
import numpy as np
import keras
import tensorflow as tf
#import shutil
from keras.utils.np_utils import to_categorical

path_prefix = './classy/'

class_list = os.listdir(path_prefix)
num_classes = len(class_list)

image_size = (224,224)


def load_dataset():
    print("getting the data in MNIST Format")
    data = []
    label = []
    for i in range(len(class_list)):
        if class_list[i]== 'miss':
            class_label = 1
        elif class_list[i]== 'nomiss':
            class_label = 0
        img_list = os.listdir(path_prefix+class_list[i]+ '/')
        for j in range(len(img_list)):
            img_name = path_prefix+class_list[i]+'/'+img_list[j]
            img = cv2.imread(img_name)
            img = cv2.resize(img,image_size)
            data.append(img)
            label.append(class_label)
    data = np.array(data,'float32')
    label = np.array(label)
    shuffle_id = np.arange(len(data))
    np.random.shuffle(shuffle_id)
    data = data[shuffle_id, :,:,:]
    label = label[shuffle_id]
    return data[:200], label[:200], data[200:], label[200:]
