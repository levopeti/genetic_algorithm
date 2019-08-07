from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras.utils import np_utils

import os
from keras.datasets import fashion_mnist, mnist

# mnist = tf.keras.datasets.mnist

X_train = None
X_test = None
y_train = None
y_test = None


def load_data():
    global X_train, X_test, y_test, y_train

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)


def fully_connected(x, size_dense, dropout, weight_decay, activation, kernel_init):

    x = Dropout(rate=dropout)(x)
    x = Dense(units=size_dense, activation=activation, kernel_regularizer=l2(weight_decay),
              kernel_initializer=kernel_init)(x)

    return x


def conv_block(x, num_conv, filters, kernel_size, activation, strides, padding, kernel_init, pool_size):

    for _ in range(num_conv):
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, strides=strides, padding=padding,
                   kernel_initializer=kernel_init)(x)

    x = MaxPooling2D(pool_size=pool_size)(x)

    return x


def define_model(weight_decay, dropout, activation, kernel_init, num_block, num_conv, num_dense,
                 size_dense, kernel_size, strides, pool_size, padding, filters):
    image_size = 28
    channels = 1
    input_shape = (image_size, image_size, channels)

    ip = Input(shape=input_shape, name='input')
    x = ip
    for _ in range(num_block):
        x = conv_block(x, num_conv, filters, kernel_size, activation, strides, padding, kernel_init, pool_size)
        filters *= 2
        pool_size = 2

    x = Flatten()(x)

    for _ in range(num_dense):
        x = fully_connected(x, size_dense, dropout, weight_decay, activation, kernel_init)

    output = Dense(10, name='predictions', activation='softmax')(x)

    model = Model(inputs=ip, outputs=output, name='full_model')
    # model.summary()

    return model


def train_model_cn(parameters, only_parameters=False):
    global X_train, X_test, y_test, y_train
    model = None

    # for key, item in parameters.items():
    #     print("{}: {}".format(key, item))
    # print()

    # Distribute the model to the defined number of GPUs
    # print("Using {} GPU(s)...".format(parameters["gpu"]))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(gpu) for gpu in parameters["gpu"])

    model_params = dict(weight_decay=parameters.get("weight_decay"),
                        dropout=parameters.get("dropout"),
                        activation=parameters.get("activation"),
                        num_block=parameters.get("num_block"),
                        num_conv=parameters.get("num_conv"),
                        num_dense=parameters.get("num_dense"),
                        size_dense=parameters.get("size_dense"),
                        kernel_size=parameters.get("kernel_size"),
                        filters=parameters.get("filters"))

    if len(parameters["gpu"]) == 1:
        model = define_model(kernel_init='he_normal', pool_size=3, strides=1, padding='same', **model_params)
    elif len(parameters["gpu"]) > 1:
        with tf.device('/cpu:0'):
            model = define_model(kernel_init='he_normal', pool_size=3, strides=1, padding='same', **model_params)
        model = multi_gpu_model(model, gpus=parameters["gpu"])

    optimizers = {
        'adam': Adam(lr=parameters["learning_rate"], amsgrad=True),
        'nadam': Nadam(lr=parameters["learning_rate"]),
        'sgd': SGD(parameters["learning_rate"], decay=0, momentum=0.9, nesterov=True),
        'rmsprop': RMSprop(lr=parameters["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.9),
    }

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.get(parameters.get("optimizer")), metrics=["accuracy"])
    # print("Finished compiling.")

    num_of_params = model.count_params()

    if only_parameters:
        return num_of_params

    # Define the callbacks
    callbacks = list()
    patience = 14

    callbacks.append(EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=patience))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.3, cooldown=2, patience=5, verbose=0, min_lr=0.000001))
    # callbacks.append(ModelCheckpoint("./weights/szakdoga.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1,
    #                                    monitor='val_loss', save_best_only=True, save_weights_only=True))
    # callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_grads=True))

    # print("Start training\n")
    if X_train is None:
        load_data()

    history = model.fit(X_train, y_train,
                        batch_size=100,
                        epochs=2000,
                        verbose=0,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    result = history.history['val_acc'][-(patience + 1)]

    # print("Process is finished on gpu:{}.".format(parameters["gpu"]))

    return result, num_of_params
