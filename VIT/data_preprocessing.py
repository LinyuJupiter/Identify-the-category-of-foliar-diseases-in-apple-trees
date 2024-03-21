import cv2
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pytorch_lightning as pl


# 定义图像处理的Dataset类
class PlantDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        # 获取图像路径和标签
        self.image_id = df['images'].values
        self.labels = df.iloc[:, :-1].values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # 返回数据集的长度
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取单个样本的图像和标签
        image_id = self.image_id[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 读取图像并进行RGB转换
        image_path = f"{self.image_dir}/{image_id}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 对图像进行数据增强
        augmented = self.transform(image=image)
        image = augmented['image']

        # 返回处理后的图像和标签
        return {'images': image, 'target': label}


# 定义数据模块类
class PlantDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, CONFIG):
        super().__init__()
        self.data_dir = data_dir
        self.CONFIG = CONFIG

        # 定义训练和测试时的图像转换
        self.train_transform = Compose([
            # 定义数据增强的操作
            A.Resize(height=self.CONFIG.img_size, width=self.CONFIG.img_size),
            A.RandomResizedCrop(height=self.CONFIG.img_size, width=self.CONFIG.img_size),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8,
                                p=0.2)
            ], p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])

        self.test_transform = Compose([
            A.Resize(height=self.CONFIG.img_size, width=self.CONFIG.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    def get_df(self, mode="train"):
        path = '../plant_dataset/'
        train_df = pd.read_csv(path + f'{mode}/{mode}_label.csv')
        # 对目标变量进行独热编码
        train = train_df.copy()
        train['labels'] = train_df['labels'].apply(lambda string: string.split(' '))
        s = list(train['labels'])
        mlb = preprocessing.MultiLabelBinarizer()
        trainx = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train.index)
        trainx['images'] = train['images']
        return trainx

    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            train_df = self.get_df(mode="train")
            val_df = self.get_df(mode="val")
            self.train_dataset = PlantDataset(df=train_df, transform=self.train_transform,
                                              image_dir=f'{self.data_dir}/train_256/images/')
            self.val_dataset = PlantDataset(df=val_df, transform=self.test_transform,
                                            image_dir=f'{self.data_dir}/val_256/images/', )
        elif stage == 'test':
            test_df = self.get_df(mode="test")
            self.test_dataset = PlantDataset(df=test_df, transform=self.test_transform,
                                             image_dir=f'{self.data_dir}/test_256/images/')

    def train_dataloader(self):
        # 返回训练数据加载器
        return DataLoader(self.train_dataset, batch_size=self.CONFIG.batch_size, shuffle=True, num_workers=4,
                          drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        # 返回验证数据加载器
        return DataLoader(self.val_dataset, batch_size=self.CONFIG.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        # 返回测试数据加载器
        return DataLoader(self.test_dataset, batch_size=self.CONFIG.batch_size, num_workers=4, persistent_workers=True)
