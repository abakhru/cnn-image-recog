import cv2
import imutils
import numpy as np
import pytesseract
from skimage.segmentation import clear_border


class ANPRImageSearch:

    def __init__(self, min_ar=4, max_ar=5, debug=False):
        # store the minimum and maximum rectangular aspect ratio
        # values along with whether or not we are in debug mode
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.debug = debug

    def debug_imshow(self, title, image, wait_key=True):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if wait_key:
                cv2.waitKey(0)
            cv2.destroyAllWindows()

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation which will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)
        self.debug_imshow("Blackhat", blackhat)

        # next, find regions in the image that are light
        square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)

        # compute the Scharr gradient representation of the blackhat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")
        self.debug_imshow("Scharr", grad_x)

        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kern)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)

        # perform a series of erosions and dilations to cleanup the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)

        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, wait_key=True)

        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, clear_border_flag=False):
        # initialize the license plate contour and ROI
        lp_cnt = None
        roi = None

        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # check to see if the aspect ratio is rectangular
            if self.min_ar <= ar <= self.max_ar:
                # store the license plate contour and extract the
                # license plate from the gray scale image and then threshold it
                lp_cnt = c
                license_plate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(license_plate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # check to see if we should clear any foreground
                # pixels that are touching the border of the image
                # (which typically, not but always, indicates noise)
                if clear_border_flag:
                    roi = clear_border(roi)

                # display any debugging information and then break
                # from the loop early since we have found the license plate region
                self.debug_imshow("License Plate", license_plate)
                self.debug_imshow("ROI", roi, wait_key=True)
                break

        # return a 2-tuple of the license plate ROI and the contour associated with it
        return roi, lp_cnt

    @staticmethod
    def build_tesseract_options(psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = f"-c tessedit_char_whitelist={alphanumeric}"
        # set the PSM mode
        options += f" --psm {psm}"
        # return the built options string
        return options

    def find_and_ocr(self, image, psm=7, clear_border_flag=False):
        # initialize the license plate text
        lp_text = None

        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with th *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lp_cnt) = self.locate_license_plate(gray, candidates,
                                                 clear_border_flag=clear_border_flag)

        # only OCR the license plate   if the license plate ROI is not empty
        if lp is not None:
            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lp_text = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)

        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return lp_text, lp_cnt
