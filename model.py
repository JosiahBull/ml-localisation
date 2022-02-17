import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pandas as pd
from PIL import Image 
from PIL.ImageDraw import Draw

# Create training data
training_image_records = pd.read_csv("training.csv")
train_image_path = os.path.join(os.getcwd(), "training")

train_images = []
train_targets = []

for index, row in training_image_records.iterrows():
    (filename, xmin, ymin, xmax, ymax) = row

    train_image_fullpath = os.path.join(train_image_path, filename)
    train_img_arr = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(train_image_fullpath, target_size=(255, 255), color_mode="grayscale"))

    xmin = round(xmin/ 255, 2)
    ymin = round(ymin/ 255, 2)
    xmax = round(xmax/ 255, 2)
    ymax = round(ymax/ 255, 2)

    train_images.append(train_img_arr)
    train_targets.append((xmin, ymin, xmax, ymax))

validation_image_records = pd.read_csv("validation.csv")
validation_image_path = os.path.join(os.getcwd(), "validation")

validation_images = []
validation_targets = []

for index, row in validation_image_records.iterrows():
    (filename, xmin, ymin, xmax, ymax) = row

    validation_image_fullpath = os.path.join(validation_image_path, filename)
    validation_img_arr = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(validation_image_fullpath, target_size=(255, 255), color_mode="grayscale"))

    xmin = round(xmin/ 255, 2)
    ymin = round(ymin/ 255, 2)
    xmax = round(xmax/ 255, 2)
    ymax = round(ymax/ 255, 2)

    validation_images.append(validation_img_arr)
    validation_targets.append((xmin, ymin, xmax, ymax))


train_images = np.array(train_images)
train_targets = np.array(train_targets)

validation_images = np.array(validation_images)
validation_targets = np.array(validation_targets)

input_shape = (255, 255, 1)
input_layer = tf.keras.layers.Input(input_shape)

model_layers = layers.experimental.preprocessing.Rescaling(1./255, name='bl_1')(input_layer)
model_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(model_layers)
model_layers = layers.MaxPooling2D(name='bl_3')(model_layers)
model_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(model_layers)
model_layers = layers.MaxPooling2D(name='bl_5')(model_layers)
model_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(model_layers)
model_layers = layers.MaxPooling2D(name='bl_7')(model_layers)
model_layers = layers.Flatten(name='bl_8')(model_layers)

model_layers = layers.Dense(128, activation='relu', name='bb_1')(model_layers)
model_layers = layers.Dense(64, activation='relu', name='bb_2')(model_layers)
model_layers = layers.Dense(32, activation='relu', name='bb_3')(model_layers)
model_layers = layers.Dense(4, activation='sigmoid', name='bb_head')(model_layers)

model = tf.keras.Model(input_layer, outputs=[model_layers])

losses = {
    "bb_head":tf.keras.losses.MSE
}

model.compile(loss=losses, optimizer="Adam", metrics=['accuracy'])

train_targets = {
    "bb_head": train_targets
}

validation_targets = {
    "bb_head": validation_targets
}

history = model.fit(
    train_images,
    train_targets,
    validation_data=(validation_images, validation_targets),
    batch_size=10,
    epochs=30,
    shuffle=True,
    verbose=1
)
