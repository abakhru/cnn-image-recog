#!env python

"""
References:
- https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
- https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt
- https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg
- https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
"""
import json
import logging
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

from cnn_image_recog.logger import LOGGER

logging.getLogger('urllib3').setLevel('ERROR')
logging.getLogger('PIL').setLevel('ERROR')
LOGGER.setLevel('INFO')
torch.manual_seed(111)
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class TorchVisionImgClassification:

    def __init__(self):
        LOGGER.debug(f'All models: {json.dumps(dir(models), indent=2, sort_keys=True)}')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.model = None
        self.batch_t = None
        self.labels = self.load_labels()
        self.resnext101_32x8d()
        self.model.eval()

    @staticmethod
    def load_labels():
        imagenet_labels = Path('imagenet_classes.txt')
        if not imagenet_labels.exists():
            res = requests.get('https://raw.githubusercontent.com/Lasagne/Recipes/master/'
                               'examples/resnet50/imagenet_classes.txt')
            imagenet_labels.write_text(res.text)
        labels = imagenet_labels.read_text().splitlines()
        return labels

    def alexnet(self):
        self.model = models.alexnet(pretrained=True)

    def resnet(self):
        # First, load the model
        self.model = models.resnet101(pretrained=True)

    def resnext101_32x8d(self):
        self.model = models.resnext101_32x8d(pretrained=True)

    def load_img(self, img, expected):
        """load image from given url"""
        img = Image.open(img)
        img_t = self.transform(img)
        self.batch_t = torch.unsqueeze(img_t, 0)
        out = self.model(self.batch_t)
        return self.predict(out, expected)

    def predict(self, out, expected):
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        LOGGER.info(f'Prediction for "{expected}" image is: {self.labels[index[0]]}; '
                    f'Confidence: {percentage[index[0]].item()}')
        return self.labels[index[0]], percentage[index[0]].item()

    def predict_top5(self, out):
        LOGGER.info(f'Top 5 predictions using "resnext101_32x8d" model')
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        LOGGER.info([(self.labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
