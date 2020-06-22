from pathlib import Path

import requests
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ImageAIBase:

    def __init__(self):
        self.base_download_url = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0'
        self.project_home = Path(__file__).parent.parent.parent.resolve()
        self.model_dir = self.project_home / 'models'
        self.out_dir = self.project_home / 'o'
        self.data_dir = self.project_home / 'data'
        self.model_dir.mkdir(exist_ok=True)
        self.out_dir.mkdir(exist_ok=True)

    def download_save_model(self, model_name):
        r = requests.get(f'{self.base_download_url}/{model_name}')
        self.project_home.joinpath('models', model_name).write_bytes(r.content)
