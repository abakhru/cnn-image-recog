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
import sys
import timeit

import cv2
from imageai.Detection import VideoObjectDetection

from cnn_image_recog.imageai import ImageAIBase
from cnn_image_recog.logger import LOGGER


class VideoObjectRecognition(ImageAIBase):

    def __init__(self, input_file='2.mp4'):
        super().__init__()
        self.detector = VideoObjectDetection()
        # https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
        self.model_file_name = 'resnet50_coco_best_v2.0.1.h5'
        self.input_file = self.data_dir / input_file
        self.output_file = self.out_dir / f'proessed_{input_file}'
        if self.output_file.exists():
            self.output_file.unlink()
        self.load_model()

    def load_model(self):
        if 'resnet' in self.model_file_name.lower():
            self.detector.setModelTypeAsRetinaNet()
        model_path = self.model_dir / self.model_file_name
        if not model_path.exists():
            self.download_save_model(self.model_file_name)
        self.detector.setModelPath(os.path.join(model_path))
        self.detector.loadModel(detection_speed='flash')  # [normal, fast, faster, fastest, flash]

    def all_objects_recog(self, ):
        detections = self.detector.detectObjectsFromVideo(input_file_path=f'{self.input_file}',
                                                          output_file_path=f'{self.output_file}',
                                                          display_object_name=True,
                                                          frames_per_second=20,
                                                          minimum_percentage_probability=30,
                                                          display_percentage_probability=True,
                                                          log_progress=True)
        LOGGER.info(f'detections: {detections}')
        # for image in detections:
        #     LOGGER.info(f'{image["name"]} => '
                        # f'{image["percentage_probability"]}: {image["box_points"]}')

    def custom_objects_recog(self):
        custom_objects = self.detector.CustomObjects(person=True, motorcycle=True)
        detections = self.detector.detectCustomObjectsFromVideo(
                input_file_path=f'{self.input_file}',
                output_file_path=f'{self.output_file}',
                custom_objects=custom_objects,
                frames_per_second=20,
                log_progress=True,
                display_object_name=True,
                display_percentage_probability=True,
                minimum_percentage_probability=30)
        LOGGER.info(f'detections: {detections}')

    def live_video_feed_objects_recog(self):
        camera = cv2.VideoCapture(0)
        video_path = self.detector.detectObjectsFromVideo(
                camera_input=camera,
                output_file_path=os.path.join(self.out_dir, "camera_detected_video"),
                frames_per_second=30, log_progress=True, minimum_percentage_probability=40)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        p = VideoObjectRecognition(input_file=sys.argv[-1])
    else:
        p = VideoObjectRecognition()
    t0 = timeit.default_timer()
    # p.all_objects_recog()
    # p.custom_objects_recog()
    p.live_video_feed_objects_recog()
    LOGGER.critical(f'Total Processing time: {timeit.default_timer() - t0}')
