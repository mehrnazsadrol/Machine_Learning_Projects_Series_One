import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers
import keras.backend as K
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D 
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

path = './game-of-deep-learning-ship-datasets/train/images/'
df = pd.read_csv('./game-of-deep-learning-ship-datasets/train/train.csv')

df['path'] = path + df['image']
categories = list(df['category'])
category = {1:'Cargo', 2:'Military', 3:'Carrier', 4:'Cruise', 5:'Tankers'}
classes = []
for c in categories: 
    classes.append(category[c])
df['classes'] = classes

test_df = pd.read_csv('./game-of-deep-learning-ship-datasets/test_ApKoW4T.csv')
test_df['path'] = path + test_df['image']


plt.figure(figsize = (15,12))
for idx,image_path in enumerate(df['path']):
    if idx==24:
        break
    plt.subplot(4,8,idx+1)
    img = Image.open(image_path)
    img = img.resize((224,224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(idx)
plt.tight_layout()
plt.show()

plt.figure(figsize = (15,12))
for idx,image_path in enumerate(test_df['path']):
    if idx==24:
        break
    plt.subplot(4,8,idx+1)
    img = Image.open(image_path)
    img = img.resize((224,224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(idx)
plt.tight_layout()
plt.show()

counts = df['classes'].value_counts()
print (counts)
plt.bar(counts.index, counts.values)
name = counts.index
plt.show()


widths, heights = [], []
for i,path in enumerate(df["path"]):
    width, height = Image.open(path).size
    widths.append(width)
    heights.append(height)
    
df["width"] = widths
df["height"] = heights

print (df['width'].describe())
print (df['height'].describe())


r,g,b = [],[],[]

for i,row in df.iterrows(): 
    my_image = cv2.imread(row['path'])
    rgb = my_image.mean(axis=(0,1))
    r.append(rgb[2])
    g.append(rgb[1])
    b.append(rgb[0])

df['red'] = r
df['green'] = g
df['blue'] = b


print (df[['red','green','blue','image']].head())



X,y = df[['path','classes']],df['classes']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

vgg_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
) 



train_generator_vgg = vgg_datagen.flow_from_dataframe(
        X_train,  # This is the source directory for training images
        x_col='path',
        y_col='classes',
        target_size=(192, 192),  # All images will be resized to 150x150
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
)

val_generator_vgg = vgg_datagen.flow_from_dataframe(
        X_test,  # This is the source directory for training images
        x_col='path',
        y_col='classes',
        target_size=(192, 192),  # All images will be resized to 150x150
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
)

inception_v3_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
) 

vgg16 = VGG16(include_top = False, input_shape = (192,192,3), weights = 'imagenet')

# training of all the convolution is set to false
for layer in vgg16.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg16.output)
predictions = Dense(5, activation='softmax')(x)

model_vgg = Model(inputs = vgg16.input, outputs = predictions)
model_vgg.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
history_vgg = model_vgg.fit(
      train_generator_vgg,
      validation_data=val_generator_vgg,
      epochs=50,
      verbose=2)

plt.figure(figsize=(15,5))
plt.plot(history_vgg.history['accuracy'])
plt.plot(history_vgg.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()