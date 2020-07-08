from imageai.Prediction.Custom import ModelTraining

from cnn_image_recog.imageai import ImageAIBase


class ResetModelTrainer(ImageAIBase):

    def __init__(self):
        super().__init__()
        self.model_trainer = ModelTraining()

    def train(self):
        self.model_trainer.setModelTypeAsResNet()
        self.model_trainer.setDataDirectory(f'{self.data_dir}')
        self.model_trainer.trainModel(num_objects=4,
                                      num_experiments=10,
                                      enhance_data=False,
                                      batch_size=32,
                                      training_image_size=100,
                                      show_network_summary=False)


if __name__ == '__main__':
    p = ResetModelTrainer()
    p.train()
