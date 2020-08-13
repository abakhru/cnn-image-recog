#!/usr/bin/env python

# Convolutional Neural Network
"""
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

This neural network is trained to differentiate the characteristics
between cats and dogs. To test, add an image of a cat or a dog to the
single prediction folder and execute the entry code (besides the fit generator function)
with the path of the new image you want to use.

pip install tensorflow==2.1.0 matplotlib keras
"""

import click as click
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras import datasets

from cnn_image_recog.imageai import ImageAIBase
from cnn_image_recog.logger import LOGGER

LOGGER.critical(f'Tensorflow Version: {tf.version.VERSION}')
LOGGER.setLevel('DEBUG')
pyplot.style.use('fivethirtyeight')


class ImageRecognitionCNN(ImageAIBase):

    def __init__(self):
        """ Initialise the CNN

        - Sequential: for initializing the artificial neural network
        """
        super().__init__()
        self.model = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.history = None

    def load_dataset(self):
        (self.train_images, self.train_labels), (self.test_images,
                                                 self.test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)
        # self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
        self.train_images, self.test_images = self.prep_pixels(self.train_images, self.test_images)

    def define_model(self):
        self.model = Sequential()
        self.model.add(
                Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same',
                       input_shape=(32, 32, 3)))
        self.model.add(BatchNormalization())
        self.model.add(
                Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(
                Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(
                Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(
                Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(
                Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                       padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        # compile model

    def define_model_1(self):
        """Build the convolution

        Layers:
        - Convolution2D: for implementing the convolution network that works with images
        - MaxPooling2D: for adding the pooling layers
        - Flatten: for converting pooled feature maps into one column,
                   that will be fed to the fully connected layer
        - Dense: that will add a fully connected layer to the neural network
        """
        LOGGER.info('32 filters or a 3x3 grid')
        first_layer = Conv2D(filters=32,
                             kernel_size=(3, 3),
                             input_shape=(32, 32, 3),
                             activation='relu')
        self.model.add(first_layer)
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        LOGGER.info('Second layer')
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        LOGGER.info('Third layer')
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        LOGGER.info('3 - Flattening')
        self.model.add(Flatten())
        LOGGER.info('4 - Full Connection, making an ANN')
        self.model.add(Dense(activation="relu", units=64))
        LOGGER.info('Binary outcome so sigmoid is being used')
        self.model.add(Dense(activation="sigmoid", units=10))
        self.model.summary()

    def compile(self):
        """Compiling the NN"""
        # self.model.compile(optimizer='adam',
        #                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                    # loss='binary_crossentropy',
        #                    metrics=['accuracy'])
        opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epoch=50):
        """Fitting the neural network for the images"""
        # self.history = self.model.fit(self.train_images,
        #                               self.train_labels,
        #                               epochs=epoch,
        #                               validation_data=(self.test_images, self.test_labels))
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                     horizontal_flip=True)
        # prepare iterator
        it_train = datagen.flow(self.train_images, self.train_labels, batch_size=64)
        # fit model
        steps = int(self.train_images.shape[0] / 64)
        self.history = self.model.fit_generator(it_train, steps_per_epoch=steps,
                                                epochs=epoch,
                                                validation_data=(self.test_images,
                                                                 self.test_labels),
                                                verbose=1)
        # evaluate model
        _, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        LOGGER.info('> %.3f' % (acc * 100.0))
        self.summarize_diagnostics()

    def summarize_diagnostics(self):
        """plot diagnostic learning curves"""
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.history.history['loss'], color='blue', label='train')
        pyplot.plot(self.history.history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
        pyplot.plot(self.history.history['val_accuracy'], color='orange', label='test')
        # save plot to file
        pyplot.savefig(f'{self.project_home / "o"}/{self.model.name}_plot.png')
        pyplot.close()

    def verify_data(self):
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        pyplot.figure(figsize=(10, 10))
        for i in range(25):
            pyplot.subplot(5, 5, i + 1)
            pyplot.xticks([])
            pyplot.yticks([])
            pyplot.grid(False)
            pyplot.imshow(self.train_images[i], cmap=pyplot.cm.ma)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            pyplot.xlabel(class_names[self.train_labels[i][0]])
        pyplot.show()

    @staticmethod
    def prep_pixels(train, test):
        """ scale pixels"""
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

    def evaluate(self):
        pyplot.plot(self.history.history['accuracy'], label='accuracy')
        pyplot.plot(self.history.history['val_accuracy'], label='val_accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.ylim([0.5, 1])
        pyplot.legend(loc='lower right')
        pyplot.savefig(fname=f'o/{self.model.name}_accuracy_history.png')
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        LOGGER.info("model, accuracy: {:5.2f}%".format(100 * test_acc))
        pyplot.close()

    def save_model(self):
        model_file_path = self.model_dir / 'my_model.h5'
        if model_file_path.exists():
            model_file_path.unlink()
        self.model.save(f'{model_file_path}')

    def load_model(self, model_file_path=None):
        if not model_file_path:
            model_file_path = self.model_dir / 'my_model.h5'
            assert model_file_path.exists() is True
        self.model = tf.keras.models.load_model(str(model_file_path))
        # self.model.summary()

    def predict(self):
        """--------- New Prediction -------------"""
        test_files = (self.data_dir / 'test' / 'cat').rglob('*.jpg')
        LOGGER.debug(f'Processing {len(list(test_files))} files')
        for test_image in test_files:
            LOGGER.info(f'Processing {test_image}...')
            _image = image.load_img(f'{test_image}', target_size=(32, 32))
            LOGGER.debug('Change to a 3 Dimensional array because it is a colour image')
            _image = image.img_to_array(_image)
            LOGGER.debug('add a forth dimension')
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
        p.load_dataset()
        # p.verify_data()
        p.define_model()
        p.compile()
        p.train(epoch=10)
        # p.evaluate()
        p.save_model()
    # if test:
    p.load_model()
    p.predict()


if __name__ == '__main__':
    main()
