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

import torch
from PIL import Image
from src.logger import LOGGER
from parameterized import parameterized
from torchvision import models
from torchvision.transforms import transforms

logging.getLogger('urllib3').setLevel('ERROR')
logging.getLogger('PIL').setLevel('ERROR')
LOGGER.setLevel('INFO')


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
        with open('imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]
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


p = TorchVisionImgClassification()


@parameterized.expand(
        [('dog.jpg', Path('data/dog.jpg'), 'Labrador retriever'),
         ('dog.jpeg', Path('data/dog.jpeg'), 'Labrador retriever'),
         ('cat.jpg', Path('data/cat.jpg'), 'tabby, tabby cat'),
         ('cat_2.jpg', Path('data/cat_2.jpg'), 'tabby, tabby cat'),
         ('rooster.jpg', Path('data/rooster.jpg'), 'maypole'),
         ('iron_chic.jpg', Path('data/iron_chic.jpg'), 'maraca')]
        # [(i.name, i, 'abc') for i in list(Path().cwd().joinpath('data').rglob('*'))]
        )
def test_images(image_name, image_path, expected):
    LOGGER.debug(f'Running image classification test for "{image_name}" file')
    assert image_path.exists() is True, f'{image_path} file not found'
    classification, confidence = p.load_img(image_path, expected)
    assert classification == expected, (f'"{expected}" classification does not match with actual "'
                                        f'{classification} classification')
