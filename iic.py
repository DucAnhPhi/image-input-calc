import datacollection as dc
import torch
import re
import cv2
import os
import numpy as np
from network import CharacterClassifier
import training_data as td


def data_collection():
    # TODO: Call the data collection functions here
    pass


def extract_labels(model):
    file_path = "SubImages"
    images = []
    sym_idx = []
    for file_name in sorted(os.listdir(file_path)):
        sym_idx.append([int(i) for i in re.findall(r'\d+', file_name)])
        img = cv2.imread(file_path + '/' + file_name)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        np_img = np.asarray(img).reshape((1, 32, 32))
        images.append(np_img)
    image_tensor = torch.Tensor(images)
    prediction = model(image_tensor)
    labels = torch.argmax(prediction, dim=1)

    eps = 1e-02

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


if __name__ == '__main__':
    data_collection()

    # TODO: Change the dimensions, when a model with real training data is used

    classifier = CharacterClassifier((1, 32, 32), [50], 19)
    classifier.load_state_dict(torch.load('hasy_model-trans.ckpt'))

    recognized = extract_labels(classifier)

    for line in recognized:
        line_str = ""
        for symbol in line:
            line_str = line_str + str(symbol) + " "
        print(line_str)
