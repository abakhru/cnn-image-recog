#!env python

"""
https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md

The object detection model (RetinaNet) supported by ImageAI can detect 80 different types of objects.
They include:

person,  bicycle,  car, motorcycle, airplane, bus, train,  truck,  boat,  traffic light,
stop_sign, oven,  toothbrush.  fire hydrant,  orange,
parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,
broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
"""
import os
import timeit

from imageai.Detection import VideoObjectDetection

from src.imageai import ImageAIBase
from src.logger import LOGGER


class VideoObjectRecognition(ImageAIBase):

    def __init__(self):
        super().__init__()
        self.detector = VideoObjectDetection()
        # https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
        self.model_file_name = 'resnet50_coco_best_v2.0.1.h5'
        self.load_model()

    def load_model(self):
        if 'resnet' in self.model_file_name.lower():
            self.detector.setModelTypeAsRetinaNet()
        model_path = self.model_dir / self.model_file_name
        if not model_path.exists():
            self.download_save_model(self.model_file_name)
        self.detector.setModelPath(os.path.join(model_path))
        self.detector.loadModel(detection_speed='fast')

    def all_objects_recog(self):
        detections = self.detector.detectObjectsFromVideo(
                input_file_path=os.path.join(self.data_dir, 'traffic.mp4'),
                output_file_path=os.path.join(self.out_dir, 'processed_traffic.mp4'),
                minimum_percentage_probability=50,
                display_percentage_probability=False,
                log_progress=True)
        for image in detections:
            LOGGER.info(f'{image["name"]} => '
                        f'{image["percentage_probability"]}: {image["box_points"]}')

    def custom_objects_recog(self):
        custom_objects = self.detector.CustomObjects(person=True)
        detections = self.detector.detectCustomObjectsFromVideo(
                custom_objects=custom_objects,
                minimum_percentage_probability=30)

        for image in detections:
            LOGGER.info(f'{image["name"]} => '
                        f'{image["percentage_probability"]}: {image["box_points"]}')


if __name__ == '__main__':
    p = VideoObjectRecognition()
    t0 = timeit.default_timer()
    p.all_objects_recog()
    # p.custom_objects_recog()
    LOGGER.critical(f'Total Processing time: {timeit.default_timer() - t0}')
