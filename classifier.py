import torch
import re
import os
from torchvision import models
import training_data as td
from torch.nn import Conv2d
import cv2
import numpy as np
from PIL import Image

"""
A classifier for handwritten math symbols 
Trained with HASY, MNIST, and our own segmented data
:author: Fenja Kollasch
"""
class MathSymbolClassifier():
    """
    Initialization
    :param model_path: Relative path to the pretrained model you want to use
    """
    def __init__(self, model_path):
        self.classifier = models.densenet201(num_classes=len(td.MATH_SYMBOLS))
        self.classifier.features.conv0 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classifier.load_state_dict(torch.load(model_path))

    """
    Classifies the images that were created during preprocessing and segmentation
    :param line_list: A list of symbols ordered as a 2d list where each list of symbols represents a line
    """
    def classify(self,  line_list):
        for line in line_list:
            line_vector = np.ndarray((len(line), 1, 32, 32))
            idx = 0
            for symbol in line:
                resized_symbol = cv.resize(symbol, (32,32))
                line_vector[idx] = resized_symbol.reshape(1, 32, 32)
            result = ''.join(self.classifier.classify(line_vector))
            print("Input:", result)
