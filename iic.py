import torch
import re
import os
from torchvision import models
import training_data as td
from torch.nn import Conv2d
import cv2
import numpy as np
from PIL import Image


class MathSymbolClassifier():
    def __init__(self, model_path):
        self.classifier = models.alexnet(num_classes=15)
        self.classifier.features[0] = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier.load_state_dict(torch.load(model_path))

    def test_classification(self):
        file_path = "SubImages"
        images = torch.zeros((20, 1, 32, 32))
        sym_idx = []
        i = 0
        for file_name in sorted(os.listdir(file_path)):
            sym_idx.append([int(i) for i in re.findall(r'\d+', file_name)])
            img = cv2.imread(file_path + '/' + file_name)
            processed = td.MyDataSet.preprocess(img)
            images[i] = processed
            i += 1

        prediction = self.classifier(images)
        labels = torch.argmax(prediction, dim=1)

        lines = []
        line = []
        current_line_idx = 0
        for i in range(len(sym_idx)):
            if current_line_idx != sym_idx[i][0]:
                lines.append(line)
                line = []
            line.append(td.MATH_SYMBOLS[labels[i].item()])
            current_line_idx = sym_idx[i][0]
        return lines

    def classify(self, imgs):
        img_tensor = torch.Tensor(imgs)

        labels = torch.argmax(self.classifier(img_tensor), axis=1)
        return [td.MATH_SYMBOLS[label] for label in labels]


if __name__ == '__main__':
    cls = MathSymbolClassifier('combined-model-92.ckpt')

    recognized = cls.test_classification()

    for line in recognized:
        line_str = ""
        for symbol in line:
            line_str = line_str + str(symbol) + " "
        print(line_str)
