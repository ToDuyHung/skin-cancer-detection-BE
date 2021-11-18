import tensorflow as tf
import numpy as np
import sys
from PIL import Image
from pathlib import Path
import gdown
import pickle
import utils

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
    

    def predict(self, input):
        if self.model:
            input_after_preprocess = self.preprocess(input)
            prediction = self.model.predict(input_after_preprocess)
            return self.get_prediction_string(prediction)

        return None

    def get_prediction_string(self, prediction):
        idx = int(np.argmax(prediction))
        return self.cancer_types[idx]


class SimpleANN(MLModel):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model('ML_models/simple_ANN.h5')

    def preprocess(self, input):
        # Resize image and convert to numpy array
        np_img = np.asarray(input.img.resize((125, 100)))

        # Normalization
        # Hard coded mean and std from jupyter notebook
        x_train_mean = 159.98101160431366
        x_train_std = 46.39439584230304
        np_img = (np_img - x_train_mean) / x_train_std

        # Reshape input to compatible with ANN model
        np_img = np_img.reshape((1, 125*100*3))
        return np_img

class MixedDataModelV1(MLModel):
    def __init__(self):
        super().__init__()
        saved_path = 'ML_models/mixed_data_model_v1/'
        model_file = saved_path + 'model.hdf5'
        binarizers_file = saved_path + 'preprocess_binarizers.pkl'

        Path(saved_path).mkdir(parents=True, exist_ok=True)
        if not Path(model_file).exists():
            url = 'https://drive.google.com/uc?id=1A-4jBVRZIbT32RMJczu_SFt9PZFy_yUx'
            gdown.download(url, model_file, quiet=False)

        if not Path(binarizers_file).exists():
            url = 'https://drive.google.com/uc?id=1-1KkQblb2lNY5YFY4vNP22vCLLf59nJr'
            gdown.download(url, binarizers_file, quiet=False)

        self.model = tf.keras.models.load_model(model_file)
        with open(binarizers_file, 'rb') as f:
            self.ageScaler, self.sexBinarizer, self.localizationBinarizer, self.labelBinarizer = pickle.load(f)
        
    def preprocess(self, input):
        img = utils.aspect_aware_resize(np.asarray(input.img), 224, 224)
        age = self.ageScaler.transform(np.array([input.age]).reshape(-1, 1))
        gender = self.sexBinarizer.transform([input.gender])
        localization = self.localizationBinarizer.transform([input.localization])
        clinicalData = np.hstack((age, gender, localization))
        return [clinicalData, np.expand_dims(img, axis=0)]