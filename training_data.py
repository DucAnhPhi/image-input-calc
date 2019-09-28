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
                '+', '-', '/', '(', ')']
SYMBOL_CODES = [
                196]

IMAGE_DIMS = (1, 32, 32)


class DataCollection(Dataset):

    def __init__(self, data_root='HASY', data_subfolder=os.path.join('classification-task', 'fold-1'),
                 train_file='train.csv', test_file='test.csv', train=True, use_hasy=True, use_mnist=True, use_own=True,
                 own_path='all_symbols', no_strokes=False):
        super(DataCollection, self).__init__()
        self.train = train
        self.data_root = data_root
        self.data_subfolder = os.path.join(self.data_root, data_subfolder)
        self.no_labels = len(MATH_SYMBOLS)-2 if no_strokes else len(MATH_SYMBOLS)
        self.img_dims = IMAGE_DIMS
        self.train_file = train_file
        self.test_file = test_file

        imgs, labels = [], []
        if use_hasy:
            imgs, labels = self.get_hasy_data()
        if use_mnist:
            self.append_mnist(imgs, labels, train)
        if use_own:
            self.append_own(imgs, labels, train, path=own_path, no_strokes=no_strokes)
        self.data = imgs
        self.targets = labels
        self.size = len(imgs)

    def get_hasy_data(self):
        if self.train:
            file = self.train_file
        else:
            file = self.test_file
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
                        imgs.append(self.preprocess(img))
                        labels.append(label_idx)
                        self.data_augmentation(img, label_idx, imgs, labels)
        return imgs, labels

    @staticmethod
    def preprocess(img, invert=True):
        return DataCollection.torch_preprocess(DataCollection.custom_preprocessing(img, invert))

    @staticmethod
    def torch_preprocess(img):
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        return preprocess(img)

    @staticmethod
    def custom_preprocessing(img, invert=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = DataCollection.resize_keep_ratio(img, size=32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        custom_preprocessing = PreProcessing()
        preprocessed = custom_preprocessing.convert_gray(img)
        preprocessed = custom_preprocessing.gaussian_blur(preprocessed)
        preprocessed = custom_preprocessing.binarize(preprocessed)
        pil_img = Image.fromarray(preprocessed)
        if invert:
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

    @staticmethod
    def data_augmentation(img, label, imgs, labels, pillow=False, invert=True):
        if not pillow:
            img = DataCollection.custom_preprocessing(img, invert)
        flip = transforms.RandomHorizontalFlip()
        rotation2 = transforms.RandomRotation(10)
        rotation3 = transforms.RandomRotation(5)
        augmented = [rotation2(img), rotation3(img), flip(img),
                     rotation2(flip(img)),flip(rotation2(img)), 
		     rotation3(flip(img)), flip(rotation3(img))]
        for augmented_img in augmented:
            imgs.append(DataCollection.torch_preprocess(augmented_img))
            labels.append(label)

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

    def append_own(self, imgs, labels, train, path='all_symbols', no_strokes=False):
        label_idx = 0
        for symbol in MATH_SYMBOLS:
            try:
                if symbol == '/':
                    symbol = 'div'
                if symbol == '(':
                    symbol = 'brckts'
                    if no_strokes:
                        label_idx -= 2
                if train:
                    full_path = os.path.join(path, symbol)
                else:
                    full_path = os.path.join(path, symbol, 'test')
                images = os.listdir(full_path)
                for image_file in images:
                    if image_file != 'test':
                        img = cv2.imread(os.path.join(full_path, image_file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        pil_img = Image.fromarray(img, 'L')
                        rotated1 = transforms.RandomRotation(10)(pil_img)
                        rotated2 = transforms.RandomRotation(5)(pil_img)
                        augmented = [pil_img, rotated1, rotated2]
                        for pic in augmented:
                            imgs.append(self.torch_preprocess(pic))
                            labels.append(label_idx)
                            if symbol == 'brckts':
                                flipped_pic = pic.transpose(Image.FLIP_LEFT_RIGHT)
                                imgs.append(self.torch_preprocess(flipped_pic))
                                labels.append(label_idx+1)

                        if symbol == '+' or symbol == '-' or symbol == 'brckts':
                            DataCollection.data_augmentation(pil_img, label_idx, imgs, labels, pillow=True, invert=False)
                        if symbol == 'brckts':
                            DataCollection.data_augmentation(flipped_pic, label_idx+1, imgs, labels, pillow=True,
                                                             invert=False)
            except FileNotFoundError:
                print("No training data for {0}. Skipping".format(symbol))
            label_idx += 1

