from keras.layers import BatchNormalization, PReLU, Input, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Dropout, Activation
from keras.models import Model
import os
import pickle
import csv
import cv2
import numpy as np
import heapq


parent_path = os.path.dirname(os.getcwd())
dataset_path = parent_path + "/dataset/"
test_data_path = parent_path + "/dataset/test1/"

# bn + prelu
def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

# build baseline model

def build_model(out_dims, input_shape=(128, 128, 3)):
    inputs_dim = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs_dim)
    x = bn_prelu(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)

    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x_flat = Flatten()(x)

    fc1 = Dense(512)(x_flat)
    fc1 = bn_prelu(fc1)

    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=inputs_dim, outputs=fc2)

    return model

label_list_data = pickle.load(open('onhot_label_list.dat','rb'))

label_list = label_list_data[0:len(label_list_data)]

model = build_model(100)
model.load_weights("base-model.h5")

test_result_file = open(dataset_path + "test_result.csv", 'w', encoding='utf-8')
writer = csv.writer(test_result_file)
writer.writerow(['filename','label'])

for test_image_name in os.listdir(test_data_path):

    imgAbsPath = test_data_path + test_image_name
    image = cv2.imread(imgAbsPath)

    resized_image = cv2.resize(image, (128, 128))

    predict_label = model.predict(np.expand_dims(resized_image, axis=0))

    # 取概率最高的5个预测结果写入文件
    label_result = ''
    for i in range(0,5):
        max_value_idx = np.argmax(predict_label)

        target_label = label_list[max_value_idx]
        label_result = label_result + target_label[0]
        # print(max_value_idx)

        # max_value = predict_label[0][max_value_idx]
        # print(max_value)

        predict_label[0][max_value_idx] = -1

    print(label_result)

    writer.writerow([test_image_name, label_result])

test_result_file.close()
