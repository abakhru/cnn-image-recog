from pathlib import Path
from unittest import TestCase

import requests
from parameterized import parameterized

from cnn_image_recog.logger import LOGGER
from cnn_image_recog.torchvision_trial import TorchVisionImgClassification


class TestTorchVisionImage(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
        cls.get_testdata()
        cls.p = TorchVisionImgClassification()

    @classmethod
    def get_testdata(cls):
        images = [
            'https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/3/38/Greyhound_Racing_2_amk.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/c/c1/Six_weeks_old_cat_%28aka%29.jpg',
            'https://upload.wikimedia.org/wikipedia/commons/c/c1/Cat_Sphynx._img_025.jpg'
            ]
        for i in images:
            t = cls.data_dir / i.split('/')[-1]
            if not t.exists():
                res = requests.get(i)
                t.write_bytes(res.content)

    @parameterized.expand(
        [('YellowLabradorLooking_new.jpg', 'Labrador retriever'),
         ('Greyhound_Racing_2_amk.jpg', 'whippet'),
         ('Cat_November_2010-1a.jpg', 'Egyptian cat'),
         ('Cat_Sphynx._img_025.jpg', 'Egyptian cat'),
         ('rooster.jpg', 'maypole'),
         ('monkey.jpg', 'macaque'),
         ('iron_chic.jpg', 'maraca')]
        )
    def test_images(self, image_name, expected):
        LOGGER.debug(f'Running image classification test for "{image_name}" file')
        image_path = self.data_dir / image_name
        assert image_path.exists() is True, f'{image_path} file not found'
        classification, confidence = self.p.load_img(image_path, expected)
        assert classification == expected, (f'"{expected}" classification does not '
                                            f'match with actual "{classification} classification')
