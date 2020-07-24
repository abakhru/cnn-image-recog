#!/usr/bin/env python
import operator
from urllib.request import urlretrieve

import gradio as gr
import numpy as np
import requests
import tensorflow as tf
from PIL import Image

# Download human-readable labels for ImageNet.
import torch
from torchvision import transforms

from cnn_image_recog.logger import LOGGER

response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Download sample images
urlretrieve("https://www.sciencemag.org/sites/default/files/styles/article_main_large/"
            "public/cc_BE6RJF_16x9.jpg?itok=nP17Fm9H", "monkey.jpg")
urlretrieve("https://www.discoverboating.com/sites/default/files/inline-images/"
            "buying-a-sailboat-checklist.jpg", "sailboat.jpg")
urlretrieve("https://external-preview.redd.it/lG5mI_9Co1obw2TiY0e-oChlXfEQY3tsRaIjpYjERqs.jpg?"
            "auto=webp&s=ea81982f44b83efbb803c8cff8953ee547624f70", "bicycle.jpg")
urlretrieve("https://www.chamaripashoes.com/blog/wp-content/"
            "uploads/2018/09/Robert-Downey-Jr..jpg", "rdj.jpg")

mobile_net = tf.keras.applications.MobileNetV2()
inception_net = tf.keras.applications.InceptionV3()
pytorch_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()


def classify_image_with_torch(im):
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im = transforms.ToTensor()(im).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(pytorch_model(im)[0], dim=0)
    final = {labels[i]: float(prediction[i]) for i in range(1000)}
    LOGGER.info(f'[PyTorch] Final label: {max(final.items(), key=operator.itemgetter(1))[0]}')
    return final


def classify_image_with_mobile_net(im):
    arr = im.reshape((-1, 224, 224, 3))
    # arr = im.reshape((-1, 299, 299, 3))
    # arr = Image.fromarray(im.astype('uint8'), 'RGB')
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    prediction = mobile_net.predict(arr).flatten()
    final = {labels[i]: float(prediction[i]) for i in range(1000)}
    LOGGER.info(f'[MobileNet] Final label: {max(final.items(), key=operator.itemgetter(1))[0]}')
    return final


def classify_image_with_inception_net(im):
    # Resize the image to
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    im = im.resize((299, 299))
    arr = np.array(im).reshape((-1, 299, 299, 3))
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    prediction = inception_net.predict(arr).flatten()
    final = {labels[i]: float(prediction[i]) for i in range(1000)}
    LOGGER.info(f'[Inception] Final label: {max(final.items(), key=operator.itemgetter(1))[0]}')
    return final


imagein = gr.inputs.Image()
# imagein = gr.inputs.Image(shape=(299, 299, 3))
label = gr.outputs.Label(num_top_classes=3)

sample_images = [
    ["monkey.jpg"],
    ["rdj.jpg"],
    ["sailboat.jpg"],
    ["bicycle.jpg"]
    ]

gr.Interface(fn=[classify_image_with_mobile_net,
                 classify_image_with_inception_net,
                 classify_image_with_torch],
             inputs=imagein,
             outputs=label,
             title="MobileNet vs. InceptionNet",
             description="Let's compare 2 state-of-the-art machine learning models"
                         "that classify images into one of 1,000 categories: MobileNet (top),"
                         "a lightweight model that has an accuracy of 0.704, vs. InceptionNet"
                         "(bottom), a much heavier model that has an accuracy of 0.779.",
             examples=sample_images).launch()
