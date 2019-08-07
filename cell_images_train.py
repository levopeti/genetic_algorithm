import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, AvgPool2D, Flatten, GlobalAveragePooling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import random
import os
from PIL import Image
import time
import cv2

img_size = 394
train_data = None
eval_data = None
limit = -1


def load_data():
    global train_data, eval_data
    positive_images = []
    negative_images = []

    start = time.time()

    for file in os.listdir("../../cell_images/Parasitized/"):
        try:
            im_frame = Image.open("../../cell_images/Parasitized/" + file)
        except OSError:
            continue
        np_frame = np.array(im_frame) / 255
        # np_frame = cv2.resize(np_frame, (img_size, img_size))
        shape = np_frame.shape
        np_img = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        np_img[0, :shape[0], :shape[1], :] = np_frame
        # np_img[0] = np_frame
        positive_images.append(np_img)

        if len(positive_images) == limit:
            break

    for file in os.listdir("../../cell_images/Uninfected/"):
        try:
            im_frame = Image.open("../../cell_images/Uninfected/" + file)
        except OSError:
            continue
        np_frame = np.array(im_frame) / 255
        # np_frame = cv2.resize(np_frame, (img_size, img_size))
        shape = np_frame.shape
        np_img = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        np_img[0, :shape[0], :shape[1], :] = np_frame
        # np_img[0] = np_frame
        negative_images.append(np_img)

        if len(negative_images) == limit:
            break

    random.shuffle(positive_images)
    eval_pos_ratio = len(positive_images) // 10
    train_pos_img = positive_images[eval_pos_ratio:]
    eval_pos_img = positive_images[:eval_pos_ratio]
    train_pos_label = to_categorical(np.ones(len(train_pos_img)), 2)
    eval_pos_label = to_categorical(np.ones(len(eval_pos_img)), 2)
    train_pos = list(zip(train_pos_img, train_pos_label))
    eval_pos = list(zip(eval_pos_img, eval_pos_label))

    random.shuffle(negative_images)
    eval_neg_ratio = len(negative_images) // 10
    train_neg_img = negative_images[eval_neg_ratio:]
    eval_neg_img = negative_images[:eval_neg_ratio]
    train_neg_label = to_categorical(np.zeros(len(train_neg_img)), 2)
    eval_neg_label = to_categorical(np.zeros(len(eval_neg_img)), 2)
    train_neg = list(zip(train_neg_img, train_neg_label))
    eval_neg = list(zip(eval_neg_img, eval_neg_label))

    train_data = train_pos + train_neg
    random.shuffle(train_data)

    eval_data = eval_pos + eval_neg
    random.shuffle(eval_data)

    print("Load data time: {}s.".format(time.time() - start))


def build_model():
    print("Build model")
    channels = 3
    input_shape = (img_size, img_size, channels)

    ip = Input(shape=input_shape, name='input')
    x1 = Conv2D(filters=64, kernel_size=2, activation="relu", padding="same")(ip)
    x2 = Conv2D(filters=64, kernel_size=5, activation="relu", padding="same")(ip)
    x = concatenate([x1, x2], axis=3)
    x = MaxPool2D(pool_size=2)(x)

    x1 = Conv2D(filters=64, kernel_size=2, activation="relu", padding="same")(x)
    x2 = Conv2D(filters=64, kernel_size=5, activation="relu", padding="same")(x)
    x = concatenate([x1, x2], axis=3)
    x = MaxPool2D(pool_size=2)(x)

    x1 = Conv2D(filters=64, kernel_size=2, activation="relu", padding="same")(x)
    x2 = Conv2D(filters=64, kernel_size=5, activation="relu", padding="same")(x)
    x = concatenate([x1, x2], axis=3)
    # x = MaxPool2D(pool_size=2)(x)

    x = GlobalAveragePooling2D()(x)

    # x = Conv2D(filters=64, kernel_size=3, activation="relu", padding="valid")(x)
    # x = MaxPool2D(pool_size=2)(x)
    # x = Conv2D(filters=64, kernel_size=3, activation="relu", padding="valid")(x)
    # x = MaxPool2D(pool_size=2)(x)
    # x = Flatten()(x)

    x = Dense(64, activation='relu')(x)

    output = Dense(2, name='predictions', activation='softmax')(x)

    model = Model(inputs=ip, outputs=output, name='full_model')
    model.summary()

    return model


def data_generator(data, batch_size):
    imgs = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
    labels = np.zeros((batch_size, 2))
    batch_index = 0

    while True:
        for img, label in data:
            imgs[batch_index, :, :, :] = img
            labels[batch_index, :] = label
            batch_index += 1

            if batch_index == batch_size:
                yield (imgs, labels)
                imgs = np.zeros((batch_size, img_size, img_size, 3))
                labels = np.zeros((batch_size, 2))
                batch_index = 0


def train_model():
    global train_data, eval_data
    batch_size = 4
    model = build_model()

    print("Compile model")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, amsgrad=True), metrics=["accuracy"])

    print("Train model on {} samples and validate on {} samples.".format(len(train_data), len(eval_data)))
    callbacks = list()
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.3, cooldown=2, patience=10, verbose=1, min_lr=0.000001))
    callbacks.append(TensorBoard(log_dir="./"))

    train_gen = data_generator(train_data, batch_size)
    eval_gen = data_generator(eval_data, batch_size)
    history = model.fit_generator(generator=train_gen,
                                  validation_data=eval_gen,
                                  steps_per_epoch=100,
                                  validation_steps=100,
                                  epochs=1000,
                                  callbacks=callbacks)


if __name__ == "__main__":
    load_data()
    train_model()
