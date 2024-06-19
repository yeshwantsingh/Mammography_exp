#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Activation
from keras.applications import MobileNetV2, EfficientNetB2, ResNet50, Xception
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 

from sklearn.model_selection import StratifiedKFold



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
filenames = ['training10_{}/training10_{}.tfrecords'.format(i, i) for i in range(5)]

# Load data
images = []
labels = []

feature_dictionary = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_normal': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example, feature_dictionary=feature_dictionary):
    parsed_example = tf.io.parse_example(example, feature_dictionary)
    return parsed_example


def read_data(filename):
    full_dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    full_dataset = full_dataset.cache()
    
    full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    for image_features in full_dataset:
        image = image_features['image'].numpy()
        image = tf.io.decode_raw(image_features['image'], tf.uint8)
        image = tf.reshape(image, [299, 299])        
        image = image.numpy()
        image = cv2.resize(image, (100, 100))
        image = cv2.merge([image, image, image])
        images.append(image)
        labels.append(image_features['label_normal'].numpy())


for file in filenames:
    read_data(file)
    
X = np.array(images)
y = np.array(labels)


# Read numpy arrays data

with open('cv10_data/cv10_data.npy', 'rb') as f:
    X_val = np.load(f)

with open('cv10_data/cv10_labels.npy', 'rb') as f:
    y_val = np.load(f)

with open('test10_data/test10_data.npy', 'rb') as f:
    X_test = np.load(f)

with open('test10_data/test10_labels.npy', 'rb') as f:
    y_test = np.load(f)


def to_3_channel(img):
    img = cv2.resize(img, (100, 100))
    return cv2.merge((img, img, img))


X_val = np.array([to_3_channel(x) for x in X_val])
X_test = np.array([to_3_channel(x) for x in X_test])

X = np.concatenate((X, X_val), axis=0)
y = np.concatenate((y, y_val), axis=0)


# Function to add noise to an image
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss * 255, 0, 255)
    return noisy.astype(np.uint8)

# Data augmentation with noise
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_noise  # Add noise function
)



# MobileNetV2
def get_mobilenet():
    base_model = MobileNetV2(input_shape=(100, 100, 3), weights='imagenet', include_top=False)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model


# EfficientNetB2
def get_efficient_net():
    base_model = EfficientNetB2(input_shape=(100, 100, 3), weights='imagenet', include_top=False)
    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    for layer in base_model.layers:
        layer.trainable = False
    return model



# ResNet50
def get_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Replace Flatten with GlobalAveragePooling2D for better performance
    x = Dense(256, activation='relu')(x)  # Add a dense layer with 256 units
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the layers of the base model to prevent updating their weights during training
    for layer in base_model.layers:
        layer.trainable = False
    
    return model


# XceptionNet
def get_xception_net():
    base_model = Xception(input_shape=(100, 100, 3), weights='imagenet', include_top=False)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_model(model):
    if model == 'mobilenet':
        return get_mobilenet()
    elif model == 'efficientnet':
        return get_efficient_net()
    elif model == 'resnet':
        return get_resnet()
    else:
        return get_xception_net()


# Callbacks
def get_callbacks(model_name):
    c1 = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.001
    )
    
    c2 = ModelCheckpoint(
        filepath='./' + model_name + f'/{model_name}.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    return [c1, c2]


# Stratified k-fold cross-validation
def run_k_fold_cv(model_name, n_folds, n_epochs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    
    for train_index, val_index in skf.split(X, y):
        x_t, x_v = X[train_index], X[val_index]
        y_t, y_v = y[train_index], y[val_index]
        
        model = get_model(model_name)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', 'AUC'])
        
        history = model.fit(datagen.flow(x_t, y_t, batch_size=256),
                            validation_data=(x_v, y_v),
                            epochs=n_epochs,
                            callbacks=get_callbacks(model_name))
        
        # Serialize the model
        # model.save(f'model_fold_{fold_no}.h5')
        
        # Save history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'./{model_name}/history_fold_{fold_no}.csv', index=False)
        
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {scores}')

        # Save Score on test split
        score_df = pd.DataFrame(score)
        score_df.to_csv(f'./{model_name}/score_fold_{fold_no}.csv', index=False)
        
        fold_no += 1



if __name__ == '__main__':
    for model in ['mobilenet', 'efficientnet', 'resnet', 'xception']:
        run_k_fold_cv(model, 5, 50)
