import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from kviz.dense import DenseGraph


def test_regression_simple():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    Y = np.array([2 * x[0] + 3 * x[1] + 4 for x in X])

    ACTIVATION = "relu"
    model = keras.models.Sequential()
    model.add(layers.Dense(4, input_shape=(2, ), activation=ACTIVATION))
    model.add(layers.Dense(4, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))

    optimizer = keras.optimizers.RMSprop(0.025)

    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])

    model.fit(X, Y, batch_size=4, epochs=100)

    dg = DenseGraph(model)
    dg.animate_regression(X, filename="test_regression_simple")
