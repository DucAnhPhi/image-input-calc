import torch
import re
import cv2
import os
from network import CharacterClassifier
import training_data as td


class MathSymbolClassifier():
    def __init__(self, model_path):
        self.classifier = CharacterClassifier(td.IMAGE_DIMS, [50], len(td.MATH_SYMBOLS))
        self.classifier.load_state_dict(torch.load(model_path))

    def test_classification(self):
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
        prediction = self.classifier(image_tensor)
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
    cls = MathSymbolClassifier('hasy_model-02.ckpt')

    recognized = cls.test_classification()

    for line in recognized:
        line_str = ""
        for symbol in line:
            line_str = line_str + str(symbol) + " "
        print(line_str)
