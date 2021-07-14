"""
Copyright 2021 Lance Galletti

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, Normalize, ListedColormap
from networkx import DiGraph, set_node_attributes
from networkx.drawing.nx_agraph import to_agraph
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import re


def create_colormap(hex_color_string, N=25, step=51):
    """

        Parameters:
            hex_color_string: str.
                A string in format "#ffffff".
            N: int.
                should be within [1, 256]. The bigger N is, the more color the colormap will contain.
            step: int.
                Controls the range of the color map; the bigger step is, the bigger the range.
                A step that is either too big or too small might cause problems.

        Returns: a matplotlib colormap

    """
    hex_color_string = hex_color_string.lstrip('#')
    r, g, b = tuple(int(hex_color_string[i: i + 2], 16) for i in (0, 2, 4))

    left_r = max(0, r - step)
    right_r = min(255, r + step)
    left_g = max(0, g - step)
    right_g = min(255, g + step)
    left_b = max(0, b - step)
    right_b = min(255, b + step)

    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(left_r / right_r, 1, N)
    vals[:, 1] = np.linspace(left_g / right_g, 1, N)
    vals[:, 2] = np.linspace(left_b / right_b, 1, N)
    newcmp = ListedColormap(vals)
    return newcmp


class DenseGraph():
    """
        Class for creating and rendering visualization of Keras
        Sequential models with only dense layers

        Attributes:
            model : tf.keras.Model
                a compiled keras sequential model with only dense layers

        Methods:
            get_graph : returns nx.DiGraph
                DiGraph is computed based on the model provided
                you can access this attribute using this method

            set_graph : takes nx.DiGraph
                DiGraph is computed based on the model provided
                you can set this attribute to a modified DiGraph
                using this method

            render :
                Can print the network architecture or, if input
                is provided, show a GIF of the activations of each
                Neuron based on the input provided.

    """

    def __init__(self, model, input_color='#3498db', inner_color='#2ecc71', output_color='#3498db',
                 edge_clr='#B20000', input_shape='circle', inner_shape='circle', output_shape='circle'):
        """

            Parameters:
                model: A keras model.
                    Since our class is called DenseGraph(), the keras model should only contain dense layers.
                input_color: str.
                    The color of the input layer in hex form (e.g. "#FFFFFF").
                inner_color: str.
                    The color of the inner layer(s) in hex form.
                output_color: str.
                    The color of the output layer in hex form.
                edge_clr: str.
                    The color of the edge connecting nodes of different layer. In hex form.
                input_shape: str.
                    The shape of the nodes in the input layer. Should be a valid shape (e.g. "polygon").
                    Otherwise the shape will be "box" for invalid input.
                    Check https://graphviz.org/doc/info/shapes.html for some valid shapes.
                inner_shape: str.
                    The shape of the nodes in the inner layer(s). Should be a valid shape.
                output_shape:
                    The shape of the nodes in the output layer. Should be a valid shape.
        """
        expression = r'^#(?:[0-9a-fA-F]{3}){1,2}$'
        if re.search(expression, input_color) and re.search(expression, inner_color) and \
                re.search(expression, output_color) and re.search(expression, edge_clr):
            self.input_layer_node_color = input_color
            self.inner_layer_node_color = inner_color
            self.output_layer_node_color = output_color
            self.edge_color = edge_clr
            self.input_layer_node_shape = input_shape
            self.inner_layer_node_shape = inner_shape
            self.output_layer_node_shape = output_shape
        else:
            self.input_layer_node_color = '#3498db'
            self.inner_layer_node_color = '#2ecc71'
            self.output_layer_node_color = '#3498db'
            self.edge_color = '#B20000'
            self.input_layer_node_shape = 'circle'
            self.inner_layer_node_shape = 'circle'
            self.output_layer_node_shape = 'circle'

        self.model = model
        self._graph = self._make_digraph()
        self._int_models = self._get_int_models()


    def get_graph(self):
        """
            DiGraph is computed based on the model provided \
            You can access this attribute using this method

        Returns:

            networkx.DiGraph

        """
        return self._graph


    def set_graph(self, graph=DiGraph()):
        """
            DiGraph is computed based on the model provided \
            you can set this attribute to a modified DiGraph \
            using this method

            Parameters:
                graph : networkx.Graph
                    The graph to set

            Returns:
                None
        """
        self._graph = graph
        return


    def _make_digraph(self):
        """
        Constructs the DiGraph

        TODO: colors, shapes etc could be specified by user
        """

        graph = DiGraph(bgcolor="transparent", nodesep='1', ranksep='1')

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]

            if type(layer) != Dense:
                raise ValueError("Unsupported Layer Type: %s Only Dense Layers are Supported", type(layer))

            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    graph.add_node(
                        str(l) + str(n),
                        shape=self.input_layer_node_shape,
                        color=self.input_layer_node_color,
                        label=''
                    )
                else:
                    graph.add_node(
                        str(l) + str(n),
                        shape=self.inner_layer_node_shape,
                        color=self.inner_layer_node_color,
                        label=''
                    )

                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        graph.add_node(
                            str(l + 1) + str(h),
                            shape=self.output_layer_node_shape,
                            color=self.output_layer_node_color,
                            label=''
                        )
                    graph.add_edge(
                        str(l) + str(n),
                        str(l + 1) + str(h),
                        color=self.edge_color
                    )

        return graph


    def _get_int_models(self):
        """
        Creates a list of models where each model consists of all the layers
        (with matching weights and activations) in self.model up to a specified
        layer depth. This is important to get each node activation.
        """
        intermediate_models = []

        for l in range(len(self.model.layers)):
            int_model = keras.Sequential()

            for prev_layer in range(l + 1):
                int_layer = self.model.layers[prev_layer]
                int_model.add(Dense(
                    int_layer.output_shape[1],
                    input_dim=int_layer.input_shape[1],
                    activation=int_layer.activation)
                )
                int_model.layers[prev_layer].set_weights(int_layer.get_weights())
            int_model.compile(loss=self.model.loss)
            intermediate_models.append(int_model)

        return intermediate_models


    def _snap(self, filename):
        """
            Take snapshot image of the graph
        """
        out = to_agraph(self._graph)
        out.layout(prog='dot')
        out.draw(filename + '.png')
        return np.asarray(im.open(filename + '.png'))


    def _snap_X(self, i, X, filename):
        """
            Take snapshot image of the input

            TODO:
                1. this doesn't work for X in > 2 dimension
                2. how to plot input could / should be specified by user
        """
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], s=300, facecolors='none', edgecolors=self.input_layer_node_color)
        ax.scatter(X[i, 0], X[i, 1], s=300, color=self.input_layer_node_color)
        ax.spines['bottom'].set_color(self.input_layer_node_color)
        ax.spines['left'].set_color(self.input_layer_node_color)
        ax.tick_params(axis='x', colors=self.input_layer_node_color)
        ax.tick_params(axis='y', colors=self.input_layer_node_color)
        fig.savefig(filename + '_X.png', transparent=True)
        plt.close()
        return np.asarray(im.open(filename + '_X.png'))


    def _snap_learning(self, X, Y, snap_freq, filename):
        """
            Take snapshot of input with decision boundary
        """

        # create a mesh to plot in
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        meshData = np.c_[xx.ravel(), yy.ravel()]

        # TODO catch max number of classes
        colors = np.array([x for x in 'bgrcmyk'])

        fig, ax = plt.subplots(frameon=False)
        ax.scatter(X[:, 0], X[:, 1], color=colors[Y].tolist(), s=100, alpha=.9)
        self.model.fit(X, Y, batch_size=150, epochs=snap_freq)
        Z = self.model.predict(meshData)
        Z = np.array([int(round(z[0])) for z in Z]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.5, cmap=plt.cm.Paired)
        fig.savefig(filename + '.png', transparent=True)
        plt.close()

        return np.asarray(im.open(filename + '.png'))



    def _stack_gifs(self, imgs1, imgs2, filename, duration):
        """
            Takes two lists of images and stacks each image in one list on top
            of its corresponding image in the other. Then creates a GIF of the
            list of stacked images.
        """
        stacked_imgs = []
        for i in range(len(imgs1)):
            img1 = imgs1[i]
            img2 = imgs2[i]
            wpercent = img2.shape[0] / img2.shape[1]
            proportional_height = int(img1.shape[1] * wpercent)
            im2 = im.fromarray(img2)
            img2 = np.asarray(im2.resize((img1.shape[1], proportional_height)))

            stacked = im.fromarray(np.vstack((np.asarray(img2), np.asarray(img1))))
            stacked_imgs.append(stacked)

        stacked_imgs[0].save(
            filename + '_stacked.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=stacked_imgs[1:],
            loop=0,
            duration=duration,
            transparency=255,  # prevent PIL from making background black
            disposal=2
        )

        return


    def _reset(self):
        """
            Resets the graph labels, colors, fonts
        """

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]
            for n in range(0, layer.input_shape[1]):
                if l == 0:
                    set_node_attributes(self._graph, {
                        str(l) + str(n): {
                            'label': '',
                            'fontcolor': '',
                            'style': '',
                            'color': self.input_layer_node_color
                        }})
                else:
                    set_node_attributes(self._graph, {
                        str(l) + str(n): {
                            'label': '',
                            'fontcolor': '',
                            'style': '',
                            'color': self.inner_layer_node_color
                        }})

            for h in range(0, layer.output_shape[1]):
                if l == len(self.model.layers) - 1:
                    set_node_attributes(self._graph, {
                        str(l + 1) + str(h): {
                            'label': '',
                            'fontcolor': '',
                            'style': '',
                            'color': self.output_layer_node_color
                        }})


    def animate_learning(self, X, Y, epochs=100, snap_freq=10, filename='learn', duration=1000):
        """
            Make GIF from snapshots of decision boundary at given snap_freq

            Parameters:

                X : ndarray
                    input to a Keras model
                Y : ndarray
                    classes to be learned
                epochs : int
                    number of training epochs
                snap_freq : int
                    number of epochs after which to take a snapshot
                filename : str
                    name of file to save as GIF
                duration : int
                    duration in ms between images in GIF

            Returns:

                The model after learning
        """

        images = []
        for i in range(int(epochs / snap_freq)):
            images.append(im.fromarray(self._snap_learning(X, Y, snap_freq, filename)))

        images[0].save(
            filename + '.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration,
            transparency=255,  # prevent PIL from making background black
            disposal=2
        )
        return self.model


    def render(self, X=None, filename='graph', duration=1000):
        """
        Render visualization of a Sequential Dense keras model

        If X is not specified 'render()' will output the network architecture

        Parameters:
            X : ndarray
                input to a Keras model
            filename : str
                name of file to which visualization will be saved
            duration : int
                duration in ms between images in GIF

        Returns:
            None
        """

        if X is None:
            self._snap(filename)
            return

        network_images = []
        input_images = []

        # TODO find min / max node activation
        vmin = min([X[:, 0].min(), X[:, 1].min()])
        vmax = max([X[:, 0].max(), X[:, 1].max()])
        norm = Normalize(vmin=vmin - 1, vmax=vmax + 1)

        inner_cmap = create_colormap(self.inner_layer_node_color)
        input_cmap = create_colormap(self.input_layer_node_color)
        output_cmap = create_colormap(self.output_layer_node_color)

        predictions = [X]
        for i in range(len(self._int_models)):
            predictions.append(self._int_models[i].predict(X))
        predictions.append(self.model.predict(X))

        for i in range(len(X)):
            for l in range(len(self.model.layers)):
                layer = self.model.layers[l]

                for n in range(0, layer.input_shape[1]):
                    act = predictions[l][i][n]
                    if l == 0:
                        set_node_attributes(self._graph, {
                            str(l) + str(n): {
                                'style': 'filled',
                                'color': str(rgb2hex(input_cmap(norm(act))))
                            }})
                        if int(act) == act:
                            set_node_attributes(self._graph, {
                                str(l) + str(n): {
                                    'label': str(act)
                                }})
                    else:
                        set_node_attributes(self._graph, {
                            str(l) + str(n): {
                                'style': 'filled',
                                'color': str(rgb2hex(inner_cmap(norm(act))))
                            }})

                if l == len(self.model.layers) - 1:
                    network_images.append(self._snap(filename))
                    input_images.append(self._snap_X(i, X, filename))
                    self._reset()

                for h in range(0, layer.output_shape[1]):
                    if l == len(self.model.layers) - 1:
                        act = predictions[l + 1][i][h]
                        set_node_attributes(self._graph, {
                            str(l + 1) + str(h): {
                                'label': str(int(round(act))),
                                'style': 'filled',
                                'color': str(rgb2hex(output_cmap(norm(act))))
                            }})

                network_images.append(self._snap(filename))
                input_images.append(self._snap_X(i, X, filename))
                self._reset()
            self._reset()

        self._stack_gifs(network_images, input_images, filename, duration=duration)
        return
