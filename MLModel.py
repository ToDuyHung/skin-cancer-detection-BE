# import tensorflow as tf
import numpy as np
import sys
from PIL import Image
from pathlib import Path
# import gdown
import pickle
import utils
from HAM10000DataModule import HAM10000DataModule
import torch
import json
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import base64

def unnormalized_images(images, mean, std):
    unnorm = transforms.Normalize(-mean/std, 1/std)
    return unnorm(images)

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
            img_after_preprocess, meta_after_preprocess = self.preprocess(input)
            img_unnorm = unnormalized_images(img_after_preprocess, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
            
            input_after_preprocess = (img_after_preprocess.unsqueeze(0).to(self.device), meta_after_preprocess[0].unsqueeze(0).to(self.device))
            with torch.no_grad():
                prediction, attention_maps = self.model(input_after_preprocess, get_attention=True)
                attention_maps = attention_maps.sum(dim=1)

                att = cv2.resize(attention_maps[0].detach().cpu().numpy(), (224, 224), interpolation=cv2.INTER_CUBIC)

                # display the image
                img_unnorm = img_unnorm.permute(1,2,0)
                img_unnorm = img_unnorm.detach().cpu().numpy()
                img_background = plt.imshow(img_unnorm, alpha=1.0)
                plt.axis('off')
                plt.imshow(att, cmap='jet', alpha=0.5, extent=img_background.get_extent())
                plt.savefig('/home/duyhung/Documents/skin-cancer-detection-BE/attention_img/test.jpg', bbox_inches='tight')
                plt.close()
                with open("/home/duyhung/Documents/skin-cancer-detection-BE/attention_img/test.jpg", "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode("utf-8") 
                    b64_string = 'data:image/jpeg;base64,' + b64_string
                    print(b64_string[:50])
            return [self.get_prediction_string(prediction), b64_string]

        return None

    def get_prediction_string(self, prediction):
        labels = list(self.config["label2id"].keys())
        softmax = torch.nn.Softmax(dim=0)
        probabilities = torch.mul(softmax(prediction[0]), 100).tolist()
        prob_dict = dict(zip(labels, probabilities))
        max_val, second_max = sorted(prob_dict.values())[-1], sorted(prob_dict.values())[-2]
        max_key = list(prob_dict.keys())[list(prob_dict.values()).index(max_val)]
        second_key = list(prob_dict.keys())[list(prob_dict.values()).index(second_max)]
        max_val = round(max_val, 2)
        second_max = round(second_max, 2)
        print(max_key, second_key)
        res = [
            { 'label': max_key, 'value': str(max_val)},
            { 'label': 'df', 'value': str(second_max)},
            { 'label': 'other', 'value' : str(round(100 - max_val - second_max, 2))}
        ]
        print(res)
        return res


class SimpleANN(MLModel):
    def __init__(self):
        super().__init__()
        # self.model = tf.keras.models.load_model('ML_models/simple_ANN.h5')

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
        model_path = "ML_models/kfold_500epoch/"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path + "best_model.pth", map_location=self.device)
        self.config = json.load(open(model_path + "config.json", "r"))
        self.model.eval()

        datamodule = HAM10000DataModule(data_path='ML_models/datacsv',
                          img_size = 224,
                          use_meta=True,
						  train_mel = False,
						  use_ab = False, use_kfold = True)
        datamodule.setup()
        self.ageScaler = datamodule.encoder['age']
        self.sexBinarizer = datamodule.encoder['sex']
        self.localizationBinarizer = datamodule.encoder['localization']
        
    def preprocess(self, input):
        img_size = 224
        val_trans = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        img = val_trans(input.img)
        
        age = self.ageScaler.transform(np.array([input.age]).reshape(-1, 1))
        np.nan_to_num(age, copy=False)
        sex = self.sexBinarizer.transform(np.array([input.gender]).reshape(-1, 1)).toarray()
        localization = self.localizationBinarizer.transform(np.array([input.localization]).reshape(-1, 1)).toarray()
        print(age, sex, localization)
        metadata = torch.tensor(np.concatenate([age, sex, localization], axis=1), dtype=torch.float32)

        # img = img.unsqueeze(0).to(self.device)
        # metadata = metadata[0].unsqueeze(0).to(self.device)
        return (img, metadata)