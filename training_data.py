import torch
import os
import csv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class HASY(Dataset):

    def __init__(self, data_root, data_subfolder=os.path.join('classification-task', 'fold-1'), train=True):
        super(HASY, self).__init__()
        self.train = train
        self.data_root = data_root
        self.data_subfolder = os.path.join(self.data_root, data_subfolder)
        self.__make_label_maps()
        self.no_labels = 369
        self.img_dims = (3, 32, 32)
        if train:
            imgs, labels = self.__get_data_from_file('train.csv')
        else:
            imgs, labels = self.__get_data_from_file('test.csv')
        self.data = imgs
        self.targets = labels
        self.size = imgs.shape[0]

    def __get_data_from_file(self, file):
        with open(os.path.join(self.data_subfolder, file)) as label_file:
            label_reader = csv.DictReader(label_file)
            rows = list(label_reader)
            idx = 0
            length = len(rows)
            imgs = torch.zeros((length, 3, 64, 64))
            labels = torch.zeros(length)
            for i, label in tqdm(enumerate(rows)):
                img = Image.open(os.path.join(self.data_subfolder, label['path']))
                label_id = label['symbol_id']
                imgs[idx] = self.__preprocess(img)
                labels[idx] = self.symbol_to_label[label_id]
                idx += 1
        return imgs, labels

    def __make_label_maps(self):
        self.symbol_to_label = {}
        self.label_to_symbol = {}
        self.symbol_to_latex = {}
        i = 0
        with open(os.path.join(self.data_root, 'symbols.csv')) as mapping_file:
            reader = csv.DictReader(mapping_file)
            for row in reader:
                self.symbol_to_label[row['symbol_id']] = i
                self.label_to_symbol[i] = row['symbol_id']
                self.symbol_to_latex = row['latex']

    def __preprocess(self, img):
        normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.229]
        )
        preprocess = transforms.Compose([
            transforms.Grayscale(1),   
            transforms.Resize(64),
            transforms.ToTensor(),
            normalize
        ])
        return preprocess(img)

    def get_symbol(self, idx):
        return self.label_to_symbol[idx]

    def get_character(self, idx):
        return self.symbol_to_latex[self.get_symbol(idx)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Stolen from pytorch
        img, target = self.data[index], int(self.targets[index])
        return img, target