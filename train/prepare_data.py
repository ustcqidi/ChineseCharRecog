# coding=utf-8

import os
import csv
import cv2
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.misc

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

datagen = ImageDataGenerator(
                 featurewise_center = False,             #是否使输入数据去中心化（均值为0），
                 samplewise_center  = False,             #是否使输入数据的每个样本均值为0
                 featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
                 samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
                 zca_whitening = False,                  #是否对输入数据施以ZCA白化
                 rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
                 width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                 height_shift_range = 0.2,               #同上，只不过这里是垂直
                 horizontal_flip = True,                 #是否进行随机水平翻转
                 vertical_flip = False)                  #是否进行随机垂直翻转

for line in train_data_reader:

    image_name = line[0]
    label = line[1]

    if (image_name == 'filename') or (label == 'label'):
        print("discard first row")
        continue

    imgAbsPath = train_data_path + label + '/' + image_name
    image = cv2.imread(imgAbsPath)

    resized_image = cv2.resize(image, (128, 128))

    # normed_im = np.array([(resized_image - 127.5) / 127.5])

    onhot_label_value = onhot_label_dic[label]

    train_data.append([label, resized_image, onhot_label_value])

    i = 0
    im_newshape = resized_image.reshape((1,) + resized_image.shape)
    for batch in datagen.flow(im_newshape,
                              batch_size=1):  # ,
        # save_to_dir=data_path+'aug',
        # save_prefix='aug',
        # save_format='jpg'):

        batch = batch.reshape(resized_image.shape)

        img = scipy.misc.toimage(batch)
        img.show()

        train_data.append([label, batch, onhot_label_value])

        i += 1
        if i > 2:
            break  # otherwise the generator would loop indefinitely

train_data_file.close()

random.shuffle(train_data)

pickle.dump(train_data, open('train_data.dat','wb'))
