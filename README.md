# Kviz

![PyPI](https://img.shields.io/pypi/v/kviz)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
![kviz CI](https://github.com/gallettilance/kviz/workflows/kviz%20CI/badge.svg)

Visualization library for Keras Neural Networks

```
    pip3 install kviz
```

## Requirements

On Fedora

```
    sudo dnf install python3-devel graphviz graphviz-devel
```

On Ubuntu

```
    sudo apt-get install graphviz graphviz-dev
```

## Examples

### Architecture Visualization

You can visualize the architecture of your keras model as such:

```python
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy")

    dg = Visualizer(model)
    dg.render()
```

Produces the following graph:

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/base_model.png?raw=true"/>
</p>

### Learning Animation

You can visualize the learned decision boundary of your model as such:

```python
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy")

    # Generate data that looks like 2 concentric circles
    t, _ = datasets.make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=1, random_state=1)
    X = np.array(list(filter(lambda x: x[0]**2 + x[1]**2 < 1 or x[0]**2 + x[1]**2 > 1.5, t)))
    Y = np.array([1 if x[0]**2 + x[1]**2 >= 1 else 0 for x in X])

    viz = Visualizer(model)
    viz.fit(X, Y, snap_freq=20, duration=300, batch_size=4, epochs=1000, verbose=0)
```

Which produces the following GIF:

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/circle_relu_model.gif?raw=true"/>
</p>

We can try different activation functions, network architectures, etc. to see what works
best. For example, from looking at the GIF we can see that the neural net is trying to
learn a decision boundary that is a combination of two straight lines. Clearly this is
not going to work for a circular decision boundary. We could expect to better approximate
this circular decision boundary if we had more straight lines to combine. We could
try changing the number of neurons in the hidden layer to 3 or more (to learn higher
dimensional features). This produces the following (for 4 hidden neurons):

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/circle_relu_4d_model.gif?raw=true"/>
</p>

Instead, we can try changing the activation in the hidden layer to a `custom_activation`
function that is non-linear and matches our intuition of what circles are:

```python
    def custom_activation(x):
        return x**2

    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2, activation=custom_activation))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy")

    viz = Visualizer(model)
    viz.fit(X, Y, snap_freq=20, duration=300, batch_size=4, epochs=1000, verbose=0)
```

which produces:

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/circle_square_model.gif?raw=true"/>
</p>


### Node Activation Visualization

You can visualize which nodes activate in the network as a function of a set of inputs.

#### XOR Function

```python
    model = keras.models.Sequential()
    model.add(layers.Dense(2, input_dim=2, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy")

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])
    Y = np.array([x[0]^x[1] for x in X]) # Xor function

    viz = Visualizer(model)
    viz.fit(X, Y, snap_freq=20, duration=300, batch_size=4, epochs=1000, verbose=0)
    viz.view_activations_for(X)
```

Produces the following decision boundary (visualized using `matplotlib`):

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/xor_model.png?raw=true">
</p>

And the following GIF:

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/xor_model.gif?raw=true"/>
</p>

The darker the node the higher the activation is at that node.

#### Linear Decisions Boundary

```python
    import sklearn.datasets as datasets

    model = keras.models.Sequential()
    model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
    model.add(layers.Dense(1, activation=ACTIVATION))
    model.compile(loss="binary_crossentropy")

    centers = [[.5, .5]]
    t, _ = datasets.make_blobs(n_samples=50, centers=centers, cluster_std=.1)
    X = np.array(t)
    Y = np.array([1 if x[0] - x[1] >= 0 else 0 for x in X])

    viz = Visualizer(model)
    viz.fit(X, Y, snap_freq=20, duration=300, batch_size=4, epochs=1000, verbose=0)
    viz.view_activations_for(X)
```

Produces the following decision boundary (visualized using `matplotlib`):

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/line_model.png?raw=true">
</p>

And the following GIF:

<p align="center">
    <img src="https://github.com/gallettilance/kviz/blob/master/examples/line_model.gif?raw=true"/>
</p>

At a glance you can see that the activations of the middle hidden node
results in predictions of class 0 while the activation of the left-most
and right-most hiddent nodes result in predictions of class 1.

## Release

Bump the release version in the `setup.py` file, then run:

```
    make clean
    make build
    make release
```
