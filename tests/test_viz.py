import pytest
from tensorflow import keras
from tensorflow.keras import layers

from kviz.dense import DenseGraph

def test_dense_no_input():
    ACTIVATION = "sigmoid"
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    dg = DenseGraph(model)
    dg.render(filename='test_no_input')
