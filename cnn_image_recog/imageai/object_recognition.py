#!env python

"""
https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md

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

from imageai.Detection import ObjectDetection

from cnn_image_recog.imageai import ImageAIBase
from cnn_image_recog.logger import LOGGER


class ObjectRecognition(ImageAIBase):

    def __init__(self):
        super().__init__()
        self.detector = ObjectDetection()
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

    def all_objects_recog(self):
        self.detector.loadModel(detection_speed='fast')
        detections = self.detector.detectObjectsFromImage(
                input_image=os.path.join(self.data_dir, "delhi.jpg"),
                output_image_path=os.path.join(self.out_dir, "delhi_new.jpg"),
                minimum_percentage_probability=40, display_percentage_probability=False,
                display_object_name=True
                )
        for image in detections:
            LOGGER.info(f'{image["name"]} => '
                        f'{image["percentage_probability"]}: {image["box_points"]}')

    def custom_objects_recog(self):
        self.detector.loadModel(detection_speed='fast')
        custom_objects = self.detector.CustomObjects(person=True)
        detections = self.detector.detectCustomObjectsFromImage(
                custom_objects=custom_objects,
                input_image=os.path.join(self.data_dir, "delhi.jpg"),
                output_image_path=os.path.join(self.out_dir, "delhi_custom.jpg"),
                minimum_percentage_probability=30)

        for image in detections:
            LOGGER.info(f'{image["name"]} => '
                        f'{image["percentage_probability"]}: {image["box_points"]}')


if __name__ == '__main__':
    p = ObjectRecognition()
    t0 = timeit.default_timer()
    p.all_objects_recog()
    # p.custom_objects_recog()
    LOGGER.critical(f'Total Processing time: {timeit.default_timer() - t0}')
