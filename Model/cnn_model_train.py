import os
import warnings
import pickle
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

K.set_image_data_format('channels_last')

# resizing, background eliminate, grayscale for important area features
def get_image_size():
    sample = glob('gestures/*/1.jpg')
    if not sample:
        raise FileNotFoundError("No sample gesture images found in gestures/*/1.jpg")
    img = cv2.imread(sample[0], cv2.IMREAD_GRAYSCALE)
    return img.shape


def get_num_of_classes():
    return len(glob('gestures/*'))

image_x, image_y = get_image_size()
num_classes = get_num_of_classes()

#dataset - 4800 per sign , around 50+ unique signs = 2.5 lacs picture +
def load_data():
    with open('train_images', 'rb') as f:
        raw_train = pickle.load(f)
    with open('train_labels', 'rb') as f:
        y_train = np.array(pickle.load(f), dtype=np.int32)
    with open('val_images', 'rb') as f:
        raw_val = pickle.load(f)
    with open('val_labels', 'rb') as f:
        y_val = np.array(pickle.load(f), dtype=np.int32)

#reshape or resize
    def resize_list(raw_list):
        processed = []
        for img in raw_list:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (image_x, image_y))
            processed.append(resized)
        return np.array(processed, dtype=np.float32)

    X_train = resize_list(raw_train) / 255.0
    X_val   = resize_list(raw_val)   / 255.0

    X_train = X_train.reshape(-1, image_x, image_y, 1)
    X_val   = X_val.reshape(-1, image_x, image_y, 1)

    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val, num_classes)

    return X_train, y_train, X_val, y_val

#conv2D- 3 Layers 32, 64, 128- egdes, gradients - shapes, pattern - gestures overall form
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(image_x, image_y, 1)),
        BatchNormalization(), # model stable, faster learning, layer merge na ho
        MaxPooling2D((2,2), padding='same'), #  small pexels calculate, axis, rotation,shifts, lower quality img layers increase X 3

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2), padding='same'),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2,2), padding='same'),
#multidimensional data to 1D
        Flatten(),
        Dense(256, activation='relu'), #256 neurons
        Dropout(0.4),
        Dense(num_classes, activation='softmax') #0,1 convert
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    X_train, y_train, X_val, y_val = load_data()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(X_train)

    model = build_model()
    model.summary()

    checkpoint = ModelCheckpoint(
        'cnn_model.keras', monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max'
    )
    lr_plateau = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=[checkpoint, lr_plateau, early_stop]
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {acc*100:.2f}%")
    K.clear_session()
