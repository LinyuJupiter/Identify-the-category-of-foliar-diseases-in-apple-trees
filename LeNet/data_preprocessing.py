import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import image
from sklearn import preprocessing
from CONFIG import *


# 定义函数，用于获取数据集
def get_df(mode):
    path = '../plant_dataset/'
    TRAIN_DIR = path + f'{mode}_256/images/'
    train_df = pd.read_csv(path + f'{mode}/{mode}_label.csv')
    # 对目标变量进行独热编码
    train = train_df.copy()
    train['labels'] = train_df['labels'].apply(lambda string: string.split(' '))
    s = list(train['labels'])
    mlb = preprocessing.MultiLabelBinarizer()
    trainx = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train.index)
    trainx['images'] = train['images']
    # 获取特征矩阵X和目标矩阵y
    y = np.array(trainx.drop(['images'], axis=1))
    # 加载图像数据
    train_image = []
    for i in tqdm(range(train_df.shape[0])):
        img = image.load_img(TRAIN_DIR + trainx['images'][i], target_size=(TARGET_SIZE, TARGET_SIZE))
        img = image.img_to_array(img)
        img = img / 255  # 将像素值标准化到 [0, 1] 之间
        train_image.append(img)

    X = np.array(train_image)
    end = (X.shape[0])
    new_y = y[:end]
    return X, new_y
