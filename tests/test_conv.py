import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.keras.datasets import mnist

from kviz.conv import ConvGraph


def test_conv_input():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255

    number_of_classes = 10
    Y_train = utils.to_categorical(y_train, number_of_classes)
    Y_test = utils.to_categorical(y_test, number_of_classes)

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, 5, input_shape=(28, 28, 1), activation=ACTIVATION))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 5, activation=ACTIVATION))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation=ACTIVATION))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=100, epochs=5)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    dg = ConvGraph(model)
    X = []
    for i in range(number_of_classes):
        X.append(np.expand_dims(X_train[np.where(y_train == i)[0][0]], axis=0))
    dg.render(X, filename='test_input_mnist')


def test_conv_animate():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255

    number_of_classes = 10
    Y_train = utils.to_categorical(y_train, number_of_classes)
    Y_test = utils.to_categorical(y_test, number_of_classes)

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, 5, input_shape=(28, 28, 1), activation=ACTIVATION))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 5, activation=ACTIVATION))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation=ACTIVATION))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=100, epochs=5)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    dg = ConvGraph(model)
    X = []
    for i in range(min(50, len(np.where(y_train == 0)[0]))):
        X.append(np.expand_dims(X_train[np.where(y_train == 0)[0][i]], axis=0))
    dg.animate(X, filename='test_animate_mnist')
