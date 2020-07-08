#!/usr/bin/env python

"""
Jaw Points = 0–16
Right Brow Points = 17–21
Left Brow Points = 22–26
Nose Points = 27–35
Right Eye Points = 36–41
Left Eye Points = 42–47
Mouth Points = 48–60
Lips Points = 61–67

- https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e
- https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
"""

from pathlib import Path

import cv2
import dlib

from cnn_image_recog.logger import LOGGER


class FacialRecognitionExperiments:
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
        LOGGER.debug('Load the detector')
        self.detector = dlib.get_frontal_face_detector()
        LOGGER.info('Load the predictor')
        self.predictor = dlib.shape_predictor(f"{self.data_dir}/"
                                              f"shape_predictor_68_face_landmarks.dat")

    def process_image(self):
        LOGGER.debug('read the image')
        img = cv2.imread(f"{self.data_dir}/face.png")
        LOGGER.debug('Convert image into grayscale')
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        LOGGER.debug('Use detector to find landmarks')
        faces = self.detector(gray)
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point
            LOGGER.debug('Look for the landmarks')
            landmarks = self.predictor(image=gray, box=face)
            # x = landmarks.part(27).x
            # y = landmarks.part(27).y
            # Loop through all the points
            for n in range(0, 16):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=img, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
        LOGGER.debug('show the image')
        cv2.imshow(winname="Face", mat=img)
        LOGGER.debug('Wait for a key press to exit')
        cv2.waitKey(delay=0)
        LOGGER.debug('Close all windows')
        cv2.destroyAllWindows()

    def process_video(self):
        LOGGER.debug('read the image')
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            for face in faces:
                x1 = face.left()  # left point
                y1 = face.top()  # top point
                x2 = face.right()  # right point
                y2 = face.bottom()  # bottom point
                landmarks = self.predictor(image=gray, box=face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.imshow(winname="Face", mat=frame)
            # LOGGER.debug('Exit when escape is pressed')
            if cv2.waitKey(delay=1) == 27:
                break
        LOGGER.debug("When everything done, release the video capture and video write objects")
        cap.release()
        LOGGER.debug('Close all windows')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = FacialRecognitionExperiments()
    # p.process_image()
    p.process_video()
