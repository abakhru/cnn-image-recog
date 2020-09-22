# USAGE
# python ocr_license_plate.py --input license_plates/group1
# python ocr_license_plate.py --input license_plates/group2 --clear-border
from pathlib import Path

import click
import cv2
import imutils
from imutils import paths

from anpr.src.anpr import ANPRImageSearch
from cnn_image_recog.logger import LOGGER

LOGGER.setLevel('DEBUG')


def cleanup_text(text):
    """strip out non-ASCII text so we can draw the text on the image using OpenCV"""
    return ''.join([c if ord(c) < 128 else '' for c in text]).strip()


def process_image(anpr_object, image_path, psm, clear_border):
    """load the input image from disk and resize it"""
    assert Path(image_path).exists() is True
    LOGGER.info(f'Processing file "{image_path}"')
    image = cv2.imread(f'{image_path}')
    image = imutils.resize(image, width=600)

    # apply automatic license plate recognition
    (lp_text, lp_cnt) = anpr_object.find_and_ocr(image, psm=psm, clear_border_flag=clear_border)

    # only continue if the license plate was successfully OCR'd
    if lp_text is not None and lp_cnt is not None:
        # fit a rotated bounding box to the license plate contour and
        # draw the bounding box on the license plate
        box = cv2.boxPoints(cv2.minAreaRect(lp_cnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        # compute a normal (un- rotated) bounding box for the license
        # plate and then draw the OCR'd license plate text on the image
        (x, y, w, h) = cv2.boundingRect(lp_cnt)
        cv2.putText(image, cleanup_text(lp_text), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # show the output ANPR image
        LOGGER.info(f"License number for this image '{image_path}' is : {lp_text.strip()}")
        # cv2.imshow("Output ANPR", image)
        # cv2.waitKey(0)


@click.command()
@click.option("-i", "--input", is_flag=False, default='license_plates',
              required=True, help="path to input directory of images")
@click.option('-c', '--clear-border', is_flag=True, default=False,
              help="whether or to clear border pixels before OCR'ing")
@click.option("-p", "--psm", type=int, default=7,
              help="default PSM mode for OCR'ing license plates")
@click.option('-d', '--debug', is_flag=True, default=False,
              help='whether or not to show additional visualizations')
def cli(input, clear_border, psm, debug):
    # initialize our ANPR class
    anpr = ANPRImageSearch(debug=debug)

    if Path(input).is_dir():
        LOGGER.debug(f'Input {input} is a dir path')
        # grab all image paths in the input directory
        image_paths = sorted(list(paths.list_images(input)))
        # loop over all image paths in the input directory
        for image_path in image_paths:
            process_image(anpr, image_path, psm, clear_border)
    elif Path(input).is_file():
        process_image(anpr, input, psm, clear_border)


if __name__ == "__main__":
    cli()




