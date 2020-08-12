#!env python

import os
import threading
import timeit
from multiprocessing.pool import ThreadPool

from imageai.Prediction import ImagePrediction

from cnn_image_recog import DATA_DIR
from cnn_image_recog.imageai import ImageAIBase
from cnn_image_recog.logger import LOGGER

PREDICTION = None
all_images_array = list(DATA_DIR.glob('*.jpg'))


class ImagePredictionExp(ImageAIBase):

    def __init__(self):
        super().__init__()
        self.prediction = ImagePrediction()
        global PREDICTION
        PREDICTION = self.prediction
        self.model_file_name = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
        # self.model_file_name = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        # self.model_file_name = 'DenseNet-BC-121-32.h5'
        self.load_model()

    def load_model(self):
        if 'inception' in self.model_file_name.lower():
            self.prediction.setModelTypeAsInceptionV3()
        if 'densenet' in self.model_file_name.lower():
            self.prediction.setModelTypeAsDenseNet()
        if 'resnet' in self.model_file_name.lower():
            self.prediction.setModelTypeAsResNet()
        model_path = self.model_dir / self.model_file_name
        if not model_path.exists():
            self.download_save_model(self.model_file_name)
        self.prediction.setModelPath(os.path.join(model_path))

    def predict_multiple(self):
        self.prediction.loadModel()
        results_array = self.prediction.predictMultipleImages(all_images_array,
                                                              result_count_per_image=1)
        for i in range(len(results_array)):
            predictions, percentage_probabilities = (results_array[i]["predictions"],
                                                     results_array[i]["percentage_probabilities"])
            LOGGER.info(f'{all_images_array[i].name} => '
                        f'{predictions.pop()} : {percentage_probabilities.pop()}')

    def run(self, image_path):
        self.prediction.loadModel()
        # LOGGER.debug(f'starting processing for {image_path}')
        predictions, probabilities = self.prediction.predictImage(image_input=f'{image_path}',
                                                                  result_count=1)
        LOGGER.info(f'{image_path.name} => {predictions.pop()} : {probabilities.pop()}')

    def predict_threaded(self):
        pool = ThreadPool(os.cpu_count() * 2)
        pool.map(self.run, all_images_array)
        pool.join()
        pool.close()

    @staticmethod
    def predict_threaded_2():
        prediction_thread = PredictionThread()
        prediction_thread.start()
        prediction_thread.join()


class PredictionThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        PREDICTION.loadModel()
        for image_path in all_images_array:
            predictions, probabilities = PREDICTION.predictImage(image_input=f'{image_path}',
                                                                 result_count=1)
            LOGGER.info(f'{image_path.name} => {predictions.pop()} : {probabilities.pop()}')


if __name__ == "__main__":
    p = ImagePredictionExp()
    t0 = timeit.default_timer()
    # p.predict_threaded()
    p.predict_threaded_2()
    # p.predict_multiple()
    LOGGER.critical(f'Total Processing time: {timeit.default_timer() - t0}')
