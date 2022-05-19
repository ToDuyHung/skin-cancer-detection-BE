import base64
import requests
# import tensorflow as tf
# import numpy as np
# import cv2

# ------- test API -------------------

with open("akiec.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

x = requests.post(url='http://localhost:8080/predict', json = {'img': encoded_string, \
'age': 30, 'gender': 'male', 'localization': 'back'})
# x = x.json()
print(x.text)