import torch
import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset

class HASY(Dataset):

    def __init__(self, data_root, train=True, transform=None):
        super(HASY, self).__init__()
        self.train = train
        self.transform = transform
        self.data_root = data_root
        self.label_map = {}
        imgs, labels = self.__get_full_dataset()
        self.data = torch.Tensor(imgs)
        self.labels = torch.Tensor(labels)
        self.size = len(imgs)

    def __load_splits(self):
        pass

    def __get_full_dataset(self):
        imgs = []
        labels = []
        with open(os.path.join(self.data_root, 'hasy-data-labels.csv')) as label_file:
            label_reader = csv.DictReader(label_file)
            for label in label_reader:
                img = np.asarray(cv2.imread(label['path']))
                label_id = label['symbol_id']
                imgs.append(img)
                labels.append(label_id)
                self.label_map[label_id] = label['latex']
        return imgs, labels

    def show_label(self, symbol_id):
        return self.label_map[symbol_id]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.data[item]