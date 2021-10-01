import base64
import requests
# import tensorflow as tf
# import numpy as np
# import cv2

# ------- test API -------------------

with open("akiec.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

x = requests.post(url='http://localhost:6000/skin_detection', data = {'image': encoded_string})
# x = x.json()
print(x.text)

# ------- test model ----------------
# def preprocess_image_input(input_images):
#   input_images = input_images.astype('float32')
#   output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
#   return output_ims

# model = tf.keras.models.load_model('model/skin_model')
# class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# a = cv2.imread("akiec.jpg")
# b = cv2.resize(a,(224,224), interpolation=cv2.INTER_AREA)
# b = preprocess_image_input(np.array(b))
# b = b.reshape(1,224,224,3)
# b = model.predict(b)
# print(class_names[np.argmax(b)])