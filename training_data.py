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
from preprocessing import PreProcessing

MATH_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '+']
SYMBOL_CODES = [
                196]

IMAGE_DIMS = (1, 32, 32)


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
            for i, label in enumerate(tqdm(rows)):
                label_id = int(label['symbol_id'])
                for label_idx, symbol in enumerate(SYMBOL_CODES):
                    if symbol == label_id:
                        label_idx += 10
                        img = cv2.imread(os.path.join(self.data_subfolder, label['path']))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        imgs.append(self.preprocess(img))
                        labels.append(label_idx)
                        img = self.custom_preprocessing(img)
                        flip = transforms.RandomHorizontalFlip()
                        flip_img = flip(img)
                        rotation = transforms.RandomRotation(20)
                        rot_img = rotation(img)
                        rot_flip = rotation(flip(img))
                        flip_rot = flip(rotation(img))
                        imgs.append(self.torch_preprocess(flip_img))
                        imgs.append(self.torch_preprocess(rot_flip))
                        imgs.append(self.torch_preprocess(flip_rot))
                        imgs.append(self.torch_preprocess(rot_img))
                        labels.append(label_idx)
                        labels.append(label_idx)
                        labels.append(label_idx)
                        labels.append(label_idx)
        return imgs, labels

    @staticmethod
    def preprocess(img):
        return MyDataSet.torch_preprocess(MyDataSet.custom_preprocessing(img))

    @staticmethod
    def torch_preprocess(img):
        normalize = transforms.Normalize(mean=[0.485],
                                 std=[0.229])
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #normalize
        ])
        return preprocess(img)

    @staticmethod
    def custom_preprocessing(img):
        img = MyDataSet.resize_keep_ratio(img, size=32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        custom_preprocessing = PreProcessing()
        preprocessed = custom_preprocessing.convert_gray(img)
        preprocessed = custom_preprocessing.gaussian_blur(preprocessed)
        preprocessed = custom_preprocessing.binarize(preprocessed)
        np.save('test_img.png', preprocessed)
        #preprocessed = preprocessed.reshape((1, 32, 32))
        pil_img = Image.fromarray(preprocessed)
        pil_img = PIL.ImageOps.invert(pil_img)
        return pil_img

    @staticmethod
    def resize_keep_ratio(img, size=32, interpolation=cv2.INTER_AREA):
        # get height and width of given image
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w:
            return cv2.resize(img, (size, size), interpolation)
        # get longest edge
        dif = max(h, w)
        # calculate offsets
        xOffset = int((dif-w)/2.)
        yOffset = int((dif-h)/2.)
        # generate mask with longest edge and offsets
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[yOffset:yOffset+h, xOffset:xOffset+w,
                 :] = img[:h, :w, :] = img[:h, :w, :]
        # return resized mask
        return cv2.resize(mask, (size, size), interpolation)

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
                    img = cv2.imread(os.path.join(self.data_root, symbol, image_file))
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
        self.append_own(imgs, labels)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)

    def append_mnist(self, imgs, labels, train):
        if train:
            path = 'mnist_train'
            xpath = 'mnist_x_train.npy'
        else:
            path = 'mnist_test'
            xpath = 'mnist_x_test.npy'
        mnist_data = MNIST(path, train=train, download=True)
        mnist_x = np.load(xpath)
        for i in tqdm(range(mnist_x.shape[0])):
            img = np.reshape(mnist_x[i], (32, 32))
            pil_img = Image.fromarray(img, 'L')
            pil_img = PIL.ImageOps.invert(pil_img)
            imgs.append(self.torch_preprocess(pil_img))
        for label in tqdm(mnist_data.targets):
            labels.append(label.item())

    def append_own(self, imgs, labels, path='ToClassify'):
        label_idx = 0
        for symbol in MATH_SYMBOLS:
            try:
                images = os.listdir(os.path.join(path, symbol))
                for image_file in images:
                    img = cv2.imread(os.path.join(path, symbol, image_file))
                    imgs.append(self.preprocess(img))
                    labels.append(label_idx)
            except FileNotFoundError:
                print("No training data for {0}. Skipping".format(symbol))
            label_idx += 1

