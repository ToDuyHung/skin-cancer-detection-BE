import tensorflow as tf
import numpy as np
import sys
from PIL import Image

class PretrainedModel:
    _instance = None
    def __new__(self, cfg=None, *args, **kwargs):
        if not self._instance:
            self._instance = getattr(sys.modules[__name__], cfg["model"])()

        return self._instance

class MLModel():
    def __init__(self):
        self.model = None
        self.cancer_types = [
        'Actinic keratoses', 'Basal cell carcinoma',
       'Benign keratosis-like lesions ', 'Dermatofibroma',
       'Melanocytic nevi', 'Melanoma', 'Vascular lesions'
       ]

    def preprocess(self, img: Image):
        return None
    

    def predict(self, img: Image):
        if self.model:
            img_after_preprocess = self.preprocess(img)
            prediction = self.model.predict(img_after_preprocess)
            return self.get_prediction_string(prediction)

        return None

    def get_prediction_string(self, prediction):
        idx = int(np.argmax(prediction))
        return self.cancer_types[idx]


class SimpleANN(MLModel):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model('ML_models/simple_ANN.h5')

    def preprocess(self, img: Image):
        # Resize image and convert to numpy array
        np_img = np.asarray(img.resize((125, 100)))

        # Normalization
        # Hard coded mean and std from jupyter notebook
        x_train_mean = 159.98101160431366
        x_train_std = 46.39439584230304
        np_img = (np_img - x_train_mean) / x_train_std

        # Reshape input to compatible with ANN model
        np_img = np_img.reshape((1, 125*100*3))
        return np_img