import base64
import requests
import pandas as pd
import os
import glob
import json
import cv2
import numpy as np

# ------- test API -------------------

# with open("data_img/nv/ISIC_0030038_30_female_back.jpg", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode()
# x = requests.post(url='http://localhost:8080/predict', json = {'img': encoded_string, 'age': 70, 'gender': 'male', 'localization': 'trunk'})
# # x = x.json()
# print(x.text)

# -------- test multiinput ------


df = pd.read_csv('/home/duyhung/Documents/skin-cancer-detection-BE/data_test/X_test.csv')

mean = int(df['age'].mean())
df['age'] = df['age'].fillna(mean)

df['new_path'] = [_.split('/')[-1] for _ in df['path']]
# print(df)

for key in ['vasc']:
    list_img = glob.glob(f'/home/duyhung/Documents/skin-cancer-detection-BE/ham10000_test/{key}/*.jpg')
    for img in list_img:
        value = img.split('/')[-1]
        # value = os.path.join 
        tmp = df.loc[df['new_path'] == value]
        age = [_ for _ in tmp.age][0]
        gender = [_ for _ in tmp.sex][0]
        localization = [_ for _ in tmp.localization][0]
        print(1111111111111, age, gender, localization)
        with open(os.path.join(f'/home/duyhung/Documents/skin-cancer-detection-BE/ham10000_test/{key}',value), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        x = requests.post(url='http://localhost:8080/predict', json = {'img': encoded_string, 'age': age, 'gender': gender, 'localization': localization})
        
        predict = json.loads(x.text)[0]['label']
        if predict == key:
            im = cv2.imread('/home/duyhung/Documents/skin-cancer-detection-BE/attention_img/test.jpg')
            value = value.split('.')[0]
            cv2.imwrite(os.path.join(f'/home/duyhung/Documents/skin-cancer-detection-BE/data_img/{key}', f'{value}_{age}_{gender}_{localization}.jpg'), im)
        # break
