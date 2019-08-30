import torch
import os
import csv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

MATH_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'x', 'y', 'z', '*', '-', '+', '/', '(', ')']
SYMBOL_CODES = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                113, 114, 115,
                184, 195, 196, 922, 923, 924]

IMAGE_DIMS = (1, 32, 32)


class MyDataSet(Dataset):

    def __init__(self, data_root, data_subfolder='', train_file='', test_file='', train=True):
        super(MyDataSet, self).__init__()
        self.train = train
        self.data_root = data_root
        self.data_subfolder = os.path.join(self.data_root, data_subfolder)
        self.no_labels = len(MATH_SYMBOLS)
        self.img_dims = IMAGE_DIMS
        if train:
            imgs, labels = self.__get_data_from_file(train_file)
        else:
            imgs, labels = self.__get_data_from_file(test_file)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)

    def __get_data_from_file(self, file):
        with open(os.path.join(self.data_subfolder, file)) as label_file:
            label_reader = csv.DictReader(label_file)
            rows = list(label_reader)
            imgs = []
            labels = []
            for i, label in enumerate(tqdm(rows)):
                label_id = int(label['symbol_id'])
                for label_idx, symbol in enumerate(SYMBOL_CODES):
                    if symbol == label_id:
                        rotation = transforms.RandomRotation(45)
                        color_jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
                        img = Image.open(os.path.join(self.data_subfolder, label['path']))
                        imgs.append(self.__preprocess(img))
                        imgs.append(self.__preprocess(rotation(img)))
                        imgs.append(self.__preprocess(color_jitter(img)))
                        imgs.append(self.__preprocess(color_jitter(rotation(img))))
                        labels.append(label_idx)
                        labels.append(label_idx)
                        labels.append(label_idx)
                        labels.append(label_idx)
        return imgs, labels

    def __preprocess(self, img):
        normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        preprocess = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])
        return preprocess(img)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Stolen from pytorch
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def get_symbol(self, idx):
        return SYMBOL_CODES[idx]

    def get_character(self, idx):
        return MATH_SYMBOLS[self.get_symbol(idx)]


class HASY(MyDataSet):

    def __init__(self, data_root, train=True):
        super(HASY, self).__init__(data_root, data_subfolder=os.path.join('classification-task', 'fold-1'),
                                   train_file='train.csv', test_file='test.csv', train=train)


class SegmentedImgs(MyDataSet):

    def __init__(self, data_root, train=True):
        super(SegmentedImgs, self).__init__(data_root, train_file='labels.csv', test_file='labels.csv', train=train)
