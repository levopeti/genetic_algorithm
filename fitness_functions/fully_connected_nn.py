from keras.models import Model
from keras.layers import Input, Dense, Dropout

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

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 784)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)


def define_model(num_of_hidden_layers, dropouts, size_of_layers):
    image_size = 784
    channels = 1
    input_shape = (image_size,)

    ip = Input(shape=input_shape, name='input')
    x = Dropout(rate=dropouts[0])(ip)

    for i in range(num_of_hidden_layers):
        x = Dense(units=size_of_layers[i], activation="relu")(x)
        x = Dropout(rate=dropouts[i + 1])(x)

    output = Dense(10, name='predictions', activation='softmax')(x)

    model = Model(inputs=ip, outputs=output, name='full_model')
    # model.summary()

    return model


def train_model_fc(parameters, only_parameters=False):
    global X_train, X_test, y_test, y_train
    model = None

    # for key, item in parameters.items():
    #     print("{}: {}".format(key, item))
    # print()

    # Distribute the model to the defined number of GPUs
    # print("Using {} GPU(s)...".format(parameters["gpu"]))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(str(gpu) for gpu in parameters["gpu"])

    model_params = dict(num_of_hidden_layers=parameters["num_of_hidden_layers"],
                        dropouts=parameters["dropouts"],
                        size_of_layers=parameters["size_of_layers"])

    if len(parameters["gpu"]) == 1:
        model = define_model(**model_params)
    elif len(parameters["gpu"]) > 1:
        with tf.device('/cpu:0'):
            model = define_model(**model_params)
        model = multi_gpu_model(model, gpus=parameters["gpu"])

    optimizers = {
        'adam': Adam(lr=parameters["learning_rate"], amsgrad=True),
        'nadam': Nadam(lr=parameters["learning_rate"]),
        'sgd': SGD(parameters["learning_rate"], decay=0, momentum=0.9, nesterov=True),
        'rmsprop': RMSprop(lr=parameters["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.9),
    }

    model.compile(loss='categorical_crossentropy', optimizer=optimizers["adam"], metrics=["accuracy"])
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
