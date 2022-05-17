import pytorch_lightning as pl
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset



# ---------------------------------------------------

class HAM10000DataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_size, batch_size=128, sampler_func=None, use_meta=False, train_mel=False, use_ab=False, use_kfold=False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sampler_func = sampler_func
        self.sampler = None
        #  ---------------------------------- Back up-----------------------------------------
        self.train_trans = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(.8, 1.2)),
            transforms.Resize(img_size + 20),
            transforms.RandomCrop(img_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.05),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_trans = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # --------------------------------------------------------------------------------
        # self.train_trans, self.val_trans = get_transforms(img_size)
        self.train_ds, self.val_ds = None, None
        self.id2label, self.label2id = {}, {}
        self.encoder = None
        self.use_meta = use_meta
        self.train_mel = train_mel
        self.use_ab = use_ab
        self.use_kfold = use_kfold

    def setup(self, stage=None):
        if not self.train_ds:
            
            if self.use_kfold:
                X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'), index_col=0)
                y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'), index_col=0)
                X_val = pd.read_csv(os.path.join(self.data_path, 'X_val.csv'), index_col=0)
                y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv'), index_col=0)
                
                X_train_new = pd.concat([X_train, X_val], axis=0)
                y_train_new = pd.concat([y_train, y_val], axis=0)
                X_test = pd.read_csv(os.path.join(self.data_path, 'X_test.csv'), index_col=0)
                y_test = pd.read_csv(os.path.join(self.data_path, 'y_test.csv'), index_col=0)
                
                # re-assign
                X_train, y_train, X_val, y_val = X_train_new, y_train_new, X_test, y_test
            else:

                if not self.use_ab:
                    X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'), index_col=0)
                    y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'), index_col=0)
                    X_val = pd.read_csv(os.path.join(self.data_path, 'X_val.csv'), index_col=0)
                    y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv'), index_col=0)
                else:
                    X_train = pd.read_csv(os.path.join(self.data_path, 'X_train_abcd.csv'))
                    y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'))
                    X_val = pd.read_csv(os.path.join(self.data_path, 'X_val_abcd.csv'))
                    y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv'))

                if self.train_mel:
                    X_train = pd.read_csv(os.path.join(self.data_path, 'X_train_augment_mel.csv'), index_col=0)
                    y_train = pd.read_csv(os.path.join(self.data_path, 'y_train_augment_mel.csv'), index_col=0)
                    X_val = pd.read_csv(os.path.join(self.data_path, 'X_val_augment_mel.csv'), index_col=0)
                    y_val = pd.read_csv(os.path.join(self.data_path, 'y_val_augment_mel.csv'), index_col=0)

            le = LabelEncoder()

            if not self.use_ab:
                self.encoder = {
                    'dx': le.fit(y_train['dx'].values),
                    'age': StandardScaler().fit(X_train['age'].values.reshape(-1, 1)),
                    'sex': OneHotEncoder().fit(X_train['sex'].values.reshape(-1, 1)),
                    'localization': OneHotEncoder().fit(X_train['localization'].values.reshape(-1, 1))
                }
            else:
                self.encoder = {
                'dx': le.fit(y_train['dx'].values),
                'age': StandardScaler().fit(X_train['age'].values.reshape(-1, 1)),
                'sex': OneHotEncoder().fit(X_train['sex'].values.reshape(-1, 1)),
                'localization': OneHotEncoder().fit(X_train['localization'].values.reshape(-1, 1)),
                'a': StandardScaler().fit(X_train['a'].values.reshape(-1, 1)),
                'e': StandardScaler().fit(X_train['e'].values.reshape(-1, 1)),
                'c': StandardScaler().fit(X_train['c'].values.reshape(-1, 1))
            }

            for i, class_name in enumerate(le.classes_):
                self.label2id[class_name] = str(i)
                self.id2label[str(i)] = class_name
    
            # Define Batch Sampler
            if self.sampler_func:
                self.sampler = self.sampler_func(self.train_ds)