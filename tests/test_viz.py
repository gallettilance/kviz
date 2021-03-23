import pytest
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from kviz.dense import DenseGraph

def test_dense_no_input():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    dg = DenseGraph(model)
    dg.render(filename='test_no_input')


def test_dense_input_xor():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])
    Y = np.array([x[0]^x[1] for x in X])

    history = model.fit(X, Y, batch_size=4, epochs=10000)

    colors = np.array(['b', 'g'])
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1], color=colors[Y].tolist(), s=50, alpha=0.8)

    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    meshData = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(meshData)
    Z = np.array([0 if x < .5 else 1 for x in Z])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)
    ax.axis('off')

    fig.savefig("test_model_xor.png")

    dg = DenseGraph(model)
    dg.render(input=X, filename='test_input_xor')
