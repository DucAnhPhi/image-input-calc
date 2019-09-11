import torch
import os
import csv
from PIL import Image
import PIL.ImageOps
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import preprocessing

MATH_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'x', 'y',
                '+', '(', ')']
SYMBOL_CODES = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                113, 114,
                196, 923, 924]

IMAGE_DIMS = (3, 224, 224)


class MyDataSet(Dataset):

    def __init__(self, data_root, data_subfolder='', train_file='', test_file='', train=True):
        super(MyDataSet, self).__init__()
        self.train = train
        self.data_root = data_root
        self.data_subfolder = os.path.join(self.data_root, data_subfolder)
        self.no_labels = len(MATH_SYMBOLS)
        self.img_dims = IMAGE_DIMS
        self.train_file = train_file
        self.test_file = test_file

    def get_data_from_file(self, file):
        with open(os.path.join(self.data_subfolder, file)) as label_file:
            label_reader = csv.DictReader(label_file)
            rows = list(label_reader)
            imgs = []
            labels = []
            custom_preprocessing = preprocessing.PreProcessing()
            for i, label in enumerate(tqdm(rows)):
                label_id = int(label['symbol_id'])
                for label_idx, symbol in enumerate(SYMBOL_CODES):
                    if symbol == label_id:
                        img = cv2.imread(os.path.join(self.data_subfolder, label['path']))
                        #img = custom_preprocessing.preprocess3(img)
                        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        img = Image.fromarray(img)
                        img = PIL.ImageOps.invert(img)
                        imgs.append(self.preprocess(img))
                        labels.append(label_idx)
        return imgs, labels

    @staticmethod
    def preprocess(img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Pad(4),
            transforms.Resize(224),
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
        if train:
            imgs, labels = super().get_data_from_file(self.train_file)
        else:
            imgs, labels = super().get_data_from_file(self.test_file)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)


class SegmentedImgs(MyDataSet):

    def __init__(self, data_root, train=True):
        super(SegmentedImgs, self).__init__(data_root, train_file='labels.csv', test_file='labels.csv', train=train)
        if train:
            imgs, labels = super().get_data_from_file(self.train_file)
        else:
            imgs, labels = super().get_data_from_file(self.test_file)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)


class OwnImgs(MyDataSet):

    def __init__(self, path='ToClassify'):
        super().__init__(path)
        imgs, labels = self.__get_data_from_folders()
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)

    def __get_data_from_folders(self):
        imgs, labels = [], []
        label_idx = 0
        for symbol in MATH_SYMBOLS:
            try:
                images = os.listdir(os.path.join(self.data_root, symbol))
                for image_file in images:
                    img = Image.open(os.path.join(self.data_root, symbol, image_file))
                    imgs.append(self.preprocess(img))
                    labels.append(label_idx)
            except FileNotFoundError:
                print("No training data for {0}. Skipping".format(symbol))
            label_idx += 1
        return imgs, labels


class CombinedData(MyDataSet):
    def __init__(self, data_root, train=True):
        super(CombinedData, self).__init__(data_root, data_subfolder=os.path.join('classification-task', 'fold-1'),
                                   train_file='train.csv', test_file='test.csv', train=train)
        if train:
            imgs, labels = super().get_data_from_file(self.train_file)
        else:
            imgs, labels = super().get_data_from_file(self.test_file)
        self.append_mnist(imgs, labels, train)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)

    def append_mnist(self, imgs, labels, train):
        if train:
            path = 'mnist_train'
        else:
            path = 'mnist_test'
        mnist_data = MNIST(path, train=train, download=True)
        for img in tqdm(mnist_data.data):
            pil_img = Image.fromarray(img.numpy())
            pil_img = PIL.ImageOps.invert(pil_img)
            imgs.append(super().preprocess(pil_img))
        for label in tqdm(mnist_data.targets):
            labels.append(label)


