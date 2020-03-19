#!/usr/bin/env python

# Convolutional Neural Network
"""
This neural network is trained to differentiate the characteristics
between cats and dogs. To test, add an image of a cat or a dog to the
single prediction folder and execute the entry code (besides the fit generator function)
with the path of the new image you want to use.

pip install tensorflow==2.1.0 matplotlib keras
"""

from pathlib import Path

import click as click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras import datasets, layers, models

from src.logger import LOGGER

LOGGER.critical(f'Tensorflow Version: {tf.version.VERSION}')


class ImageRecognitionCNN:

    def __init__(self):
        """ Initialise the CNN """
        self.model = models.Sequential()
        self.data_dir = Path(__file__).parent.resolve().joinpath('dataset')
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.history = None

    def get_data(self):
        (self.train_images, self.train_labels), (self.test_images,
                                                 self.test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

    def build(self):
        """Build the convolution"""
        LOGGER.info('32 filters or a 3x3 grid')
        first_layer = layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    input_shape=(32, 32, 3),
                                    activation='relu')
        self.model.add(first_layer)
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        LOGGER.info('Second layer')
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        LOGGER.info('Third layer')
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        LOGGER.info('3 - Flattening')
        self.model.add(layers.Flatten())
        LOGGER.info('4 - Full Connection, making an ANN')
        self.model.add(layers.Dense(activation="relu", units=64))
        LOGGER.info('Binary outcome so sigmoid is being used')
        self.model.add(layers.Dense(activation="sigmoid", units=10))
        self.model.summary()

    def compile(self):
        """Compiling the NN"""
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, epoch=50):
        """Fitting the neural network for the images"""
        self.history = self.model.fit(self.train_images,
                                      self.train_labels,
                                      epochs=epoch,
                                      validation_data=(self.test_images, self.test_labels))

    def verify_data(self):
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(class_names[self.train_labels[i][0]])
        plt.show()

    def evaluate(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig(fname=f'{self.model}_accuracy_history.png')
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        LOGGER.info("model, accuracy: {:5.2f}%".format(100 * test_acc))

    def save_model(self):
        saved_model_dir = Path(__file__).parent.resolve().joinpath('models')
        saved_model_dir.mkdir(exist_ok=True)
        model_file_path = saved_model_dir.joinpath('my_model.h5')
        if model_file_path.exists():
            model_file_path.unlink()
        self.model.save(f'{model_file_path}')

    def load_model(self, model_file_path=None):
        if not model_file_path:
            model_file_path = Path(__file__).parent.parent.resolve().joinpath('models',
                                                                             'my_model.h5')
            assert model_file_path.exists() is True
        self.model = tf.keras.models.load_model(str(model_file_path))
        # self.model.summary()

    def predict(self):
        """--------- New Prediction -------------"""
        test_files = Path(__file__).parent.parent.resolve().joinpath('data').glob('*.jpg')
        for test_image in test_files:
            _image = image.load_img(test_image, target_size=(32, 32))
            # LOGGER.debug('Change to a 3 Dimensional array because it is a colour image')
            _image = image.img_to_array(_image)
            # LOGGER.debug('add a forth dimension')
            _image = np.expand_dims(_image, axis=0)
            result = self.model.predict(_image, verbose=0)
            LOGGER.debug(f'Threshold of 50% to classify the image: {result}')
            if result[0][0] > 0.5:
                prediction = 'DOG'
            else:
                prediction = 'CAT'
            LOGGER.info(f"{test_image.name}: certainty of being a {prediction}")


@click.command()
@click.option('--build', is_flag=True, default=False, help='BUILDS THE MODEL')
@click.option('--test', is_flag=True, default=False, help='Test the model')
def main(build, test):
    p = ImageRecognitionCNN()
    if build:
        p.get_data()
        # p.verify_data()
        p.build()
        p.compile()
        p.train(epoch=50)
        p.evaluate()
        p.save_model()
    if test:
        p.load_model()
        p.predict()


if __name__ == '__main__':
    main()
