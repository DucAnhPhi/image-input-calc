import torch
import re
import cv2
import os
import training_data as td
from torchvision.models import alexnet
from torch.nn import Conv2d
import numpy as np

class MathSymbolClassifier:
    def __init__(self, num_classes=15, models={}):
        self.num_classes = num_classes
        self.classifiers = []
        for model in models.keys():
            num_classes = models[model]
            classifier = alexnet(num_classes=num_classes)
            classifier.features[0] = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            classifier.load_state_dict(torch.load(model, map_location="cpu"))
            self.classifiers.append(classifier)


    def classify(self, imgs, return_label_idxs=False):
        img_tensor = torch.Tensor(imgs)
        all_best_predicitions = None
        all_labels = []

        for classifier in self.classifiers:
            predicition = classifier(img_tensor).detach().numpy()
            labels = np.argmax(predicition, axis=1)
            if predicition.shape[1] == 13:
                labels = np.where(labels >= 11, labels+2, labels)
            all_labels.append(labels)
            best_predicition = np.max(predicition, axis=1)[None,:]
            if all_best_predicitions is None:
                all_best_predicitions = best_predicition
            else:
                all_best_predicitions = np.concatenate((all_best_predicitions, best_predicition.T), axis=1)
        classifier_compare = np.argmax(all_best_predicitions, axis=1)
        final_labels = all_labels[0]
        for i in range(1, len(self.classifiers)):
            final_labels = np.where(classifier_compare == i, all_labels[i], final_labels)
        if return_label_idxs:
            return final_labels
        return [td.MATH_SYMBOLS[label] for label in final_labels]
