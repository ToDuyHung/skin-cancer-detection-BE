import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Extract features from image and Normalize features
def extract_features(img, model):
    # input_shape = (224,224,3)
    # img = image.load_img(
    #             img_path,
    #             target_size=(
    #                 input_shape[0],
    #                 input_shape[1]
    #             )    
    #         )
    
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    
    return normalized_features

def read_features():
    filenames = pickle.load(
            open('model/filenames_full.pickle', 'rb')
        )
    feature_list = pickle.load(
            open('model/features-resnet_full.pickle', 'rb')
        )
    return filenames, feature_list

model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

filenames, feature_list = read_features()

neighbors = NearestNeighbors(
                n_neighbors=10, 
                algorithm='brute',
                metric='euclidean'
            ).fit(feature_list)

with open('test.jpg', 'wb') as f:
    im = Image.open(f)
im.show()
ft = extract_features(im, model)
distances, indices = neighbors.kneighbors([ft])
x = [filenames[indices[0][i]].split("/")[-2] for i in range(10)]