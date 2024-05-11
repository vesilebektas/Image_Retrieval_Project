import numpy as np
import shutil as sh
import os
from scipy.spatial.distance import cdist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Flatten

Data_location = r'C:/animals/'
Database_location = r'C:/Users/Vesile/PycharmProjects/Project/dene/'

if not os.path.isdir(Database_location):
    os.makedirs(Database_location)
count = 0

for idx, folder in enumerate(os.listdir(Data_location)):
    for image in os.listdir(os.path.join(Data_location, folder)):

        sh.copy(os.path.join(Data_location, folder, image), Database_location)
        count += 1
        if count == 10:
            count = 0
            break

mobile = mobilenet.MobileNet()
output = mobile.layers[-5].output
output = Flatten()(output)
model = Model(inputs=mobile.input, outputs=output)
model.summary()


def predict(image):
    image = load_img(image, target_size=(224,  224))

    image = img_to_array(image)
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    vec = model.predict(image,  verbose=0).squeeze()
    return vec

vec = predict('C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\cheetah\\0d29a5f5-da38-4dc5-a491-7b7b26725f07.jpg')
print('Shape :', vec.shape)
#load_img('C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\cheetah\\0d29a5f5-da38-4dc5-a491-7b7b26725f07.jpg')

loaded_image = load_img('C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\cheetah\\0d29a5f5-da38-4dc5-a491-7b7b26725f07.jpg',
                        target_size=(224, 224))

image_array = img_to_array(loaded_image)

plt.imshow(image_array/255.0)
plt.title('Cheetah Image')
plt.show()

name = []
vec = None
for idx, image in enumerate(os.listdir(Database_location)):
    #print("yol: ",image)
    name.append(image)
    if idx == 0:
        # print("istediğimiz : ",pridect(os.path.join(Database_location,image)))
        vec = predict(os.path.join(Database_location, image))
    else:
        # print("istediğimiz else: ", np.vstack([vec,pridect(os.path.join(Database_location,image))]))
        vec = np.vstack([vec, predict(os.path.join(Database_location, image))])

print(name[:5])
print("vec shape : ", vec.shape)

image_names = name[:5]

for image_name in image_names:
    image_path = os.path.join(Database_location, image_name)
    image = plt.imread(image_path)

    plt.figure()
    plt.imshow(image)
    plt.title(image_name)
    plt.show()

image1 = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\cheetah\\0d29a5f5-da38-4dc5-a491-7b7b26725f07.jpg'
image2 = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\lion\\0bd898cd-32e2-46bb-9cea-94751d78c49e.jpg'

vec1 = predict(image1)
vec2 = predict(image2)

print("vector for images1:", vec1)
print("vector for images2:", vec2)

# Önceden benzerlik hesaplama için kullanılan resimlerin özellik vektörleri
previously_used_vectors = vec

# Yeni bir resim için özellik vektörünü hesapla
new_image = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\lion\\0fdaa654-084d-466c-8038-a7c60f23cae4.jpg'
new_vector = predict(new_image)

# Yeni resmin, önceden kullanılan resimlerle benzerliğini hesapla
index = cdist(np.expand_dims(new_vector, axis=0), previously_used_vectors, 'cosine')
similar_images = index.argsort()[0]

print("Benzer resimlerin sırası:", similar_images)

similar_images_names = [name[i] for i in similar_images]
print("Benzer resimlerin adları:", similar_images_names)

fig,ax = plt.subplots(1, 4, figsize=(10, 10))
for idx, img_name in enumerate(similar_images_names[:4]):
    img = cv2.imread(os.path.join(Database_location, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    ax[idx].imshow(img)

#image = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\lion\\0a1a0702-a039-4d42-9e31-dd3039450ecc.jpg'
image = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\lion\\7db0fe11-cb4a-4577-b507-025f6421aaa7.jpg'
prediction = predict(image)

index = cdist(np.expand_dims(prediction, axis=0), vec, 'cosine')
similar_img = index.argsort()[0][1:6]

loaded_image = load_img(image, target_size=(224, 224))
plt.imshow(loaded_image)
plt.title('Test Image')
plt.show()

fig,ax = plt.subplots(1, 5, figsize=(10, 10))
for idx, img_idx in enumerate(similar_img):
    img_path = os.path.join(Database_location, name[img_idx])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    ax[idx].imshow(img)
    ax[idx].set_title(f'Similar Image {idx+1}')

plt.show()

image = 'C:\\Users\\Vesile\\PycharmProjects\\Project\\animals\\tiger\\0b224df6-4fcf-41f2-a011-fbe453802e39.jpg'
prediction = predict(image)

index = cdist(np.expand_dims(prediction, axis=0), vec, 'cosine')
similar_img = index.argsort()[0][1:6]

loaded_image = load_img(image, target_size=(224, 224))
plt.imshow(loaded_image)
plt.title('Test Image')
plt.show()

fig,ax = plt.subplots(1, 5, figsize=(10, 10))

for idx, img_idx in enumerate(similar_img):
    img_path = os.path.join(Database_location, name[img_idx])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    ax[idx].imshow(img)
    ax[idx].set_title(f'Similar Image {idx+1}')

plt.show()
