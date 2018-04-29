# coding=utf-8

import os
import csv
import cv2
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
import random

parent_path = os.path.dirname(os.getcwd())
dataset_path = parent_path + "/dataset/"
train_data_path = parent_path + "/dataset/train/"

train_data_file = open(dataset_path + "train.csv", 'w', encoding='utf-8')
label_file = open(dataset_path + "label.csv", 'w', encoding='utf-8')

### 预处理训练数据集
writer = csv.writer(train_data_file)
writer.writerow(['filename','label'])

label_writer = csv.writer(label_file)
labels = []

for label in os.listdir(train_data_path):
    # label_writer.writerow(label)
    labels.append(label)

    image_path = train_data_path + label
    for filename in os.listdir(image_path):
        writer.writerow([filename, label])

tokenizer = Tokenizer(num_words=None) #num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
tokenizer.fit_on_texts(labels)
onhot_label_mastrix = tokenizer.texts_to_matrix(labels)
onhot_label_mastrix = [ row[1:] for row in onhot_label_mastrix ]
label_cnt = labels.__len__()

# 武
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0.]
# 孝
# [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0.]
# ...

onhot_label_dic = {}
onhot_label_list = []
for i in range(0, label_cnt):
    label = labels[i]
    onhot_label = onhot_label_mastrix[i]
    onhot_label_dic[label] = onhot_label
    onhot_label_list.append([label, onhot_label, i])

train_data_file.close()
label_file.close()


pickle.dump(onhot_label_list, open('onhot_label_list.dat','wb'))


# for item in onhot_label_dic:
#     print(item)
#     print(onhot_label_dic[item])


### 准备训练数据集
train_data_file = open(dataset_path + "train.csv", 'r', encoding='utf-8')
train_data_reader = csv.reader(train_data_file)

train_data = []

for line in train_data_reader:

    image_name = line[0]
    label = line[1]

    if (image_name == 'filename') or (label == 'label'):
        print("discard first row")
        continue

    imgAbsPath = train_data_path + label + '/' + image_name
    image = cv2.imread(imgAbsPath)

    resized_image = cv2.resize(image, (128, 128))

    #normed_im = np.array([(resized_image - 127.5) / 127.5])

    onhot_label_value = onhot_label_dic[label]

    train_data.append([label, resized_image, onhot_label_value])

train_data_file.close()

random.shuffle(train_data)

pickle.dump(train_data, open('train_data.dat','wb'))