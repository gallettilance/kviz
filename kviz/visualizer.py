"""
Copyright 2024 Lance Galletti

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


# kviz only ever saves figures to disk (via fig.savefig); it never calls
# plt.show(). Force a non-interactive backend before importing pyplot so
# the library works headlessly. This is especially important on
# macOS + Python 3.13, where matplotlib's default `macosx` backend can
# SIGABRT when used alongside TensorFlow/Keras (resulting in no output GIF
# / PNG and no Python-level traceback).
import matplotlib
if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, LinearSegmentedColormap


from networkx import DiGraph, Graph, set_node_attributes
from networkx.drawing.nx_agraph import to_agraph

try:
    # Preferred import path; tends to be resolved by type checkers.
    from tensorflow import keras
except Exception:  # pragma: no cover
    # Fallback for environments that have standalone Keras installed.
    import keras  # type: ignore



COLORS = np.array(['purple', 'blue'])
tuples = list(zip(
    map(plt.Normalize(0, 1), [0, .5, 1]),
    [COLORS[0], 'white', COLORS[1]])
)
CMAP = LinearSegmentedColormap.from_list("", tuples, 100)


# Built-in Graphviz themes for NetworkArchitectureVisualizer.render().
# Themes are merged (per-key) with any explicit graph_attrs / default_node_attrs
# / default_edge_attrs that the caller passes; explicit values win.
NETWORK_ARCH_THEMES = {
    "layered": {
        "graph_attrs": {
            "rankdir": "LR",
            "splines": "polyline",
            "nodesep": "0.55",
            "ranksep": "1.05",
            "bgcolor": "white",
        },
        "node_attrs": {
            "shape": "circle",
            "style": "filled",
            "fillcolor": "white",
            "color": "#2c3e50",
            "fontcolor": "#222222",
            "fontsize": "9",
            "width": "0.22",
            "height": "0.22",
            "fixedsize": "true",
        },
        "edge_attrs": {
            "arrowsize": "0.5",
        },
    },
    "weighted": {
        "graph_attrs": {
            "rankdir": "LR",
            "splines": "spline",
            "nodesep": "0.6",
            "ranksep": "1.1",
            "bgcolor": "white",
        },
        "node_attrs": {
            "shape": "circle",
            "style": "filled",
            "fillcolor": "white",
            "color": "#2c3e50",
            "fontsize": "9",
            "width": "0.24",
            "height": "0.24",
            "fixedsize": "true",
        },
        "edge_attrs": {
            "arrowsize": "0.4",
        },
    },
    "pruned": {
        "graph_attrs": {
            "rankdir": "LR",
            "splines": "polyline",
            "nodesep": "0.55",
            "ranksep": "1.05",
            "bgcolor": "white",
        },
        "node_attrs": {
            "shape": "circle",
            "style": "filled",
            "fillcolor": "white",
            "color": "#2c3e50",
            "fontsize": "9",
            "width": "0.22",
            "height": "0.22",
            "fixedsize": "true",
        },
        "edge_attrs": {
            "arrowsize": "0.45",
        },
    },
}


def _resolve_theme(theme):
    """Return (graph_attrs, node_attrs, edge_attrs) for a theme spec.

    `theme` may be None, a string key into NETWORK_ARCH_THEMES, or a dict
    that already follows the {"graph_attrs": ..., "node_attrs": ...,
    "edge_attrs": ...} shape.
    """
    if theme is None:
        return {}, {}, {}
    if isinstance(theme, str):
        if theme not in NETWORK_ARCH_THEMES:
            raise ValueError(
                f"Unknown theme {theme!r}. "
                f"Available: {sorted(NETWORK_ARCH_THEMES)}"
            )
        spec = NETWORK_ARCH_THEMES[theme]
    elif isinstance(theme, dict):
        spec = theme
    else:
        raise TypeError(
            "theme must be None, a string theme name, or a dict; "
            f"got {type(theme).__name__}"
        )
    return (
        dict(spec.get("graph_attrs", {})),
        dict(spec.get("node_attrs", {})),
        dict(spec.get("edge_attrs", {})),
    )


class Visualizer():
    """
        Class for creating and rendering visualization of Keras
        Sequential models with only dense layers

        Attributes:
            model : tf.keras.Model
                a compiled keras sequential model with only dense layers

        Methods:
            render :
                Creates a Graph visualization of the model

            fit :
                Make GIF from snapshots of decision boundary at
                given intervals of epochs during training

            view_activations_for :
                Creates a GIF of the activations of each
                Neuron for the input provided
    """

    def __init__(self, model):
        """
        Parameters:
            model: A keras model.
                Since our class is called DenseGraph(), the keras model should only contain dense layers.
        """
        self.model = model
        self._graph = self._make_digraph()
        self._graph_original_copy = self._graph.copy()
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
        self._graph_original_copy = self._graph.copy()
        return


    def _unique_index(self, layer, node):
        return str(layer) + "_" + str(node)


    def _make_digraph(self):
        """
        Constructs the DiGraph
        """
        graph = DiGraph(nodesep='1', ranksep='1')

        for l in range(len(self.model.layers)):
            layer = self.model.layers[l]

            if type(layer) is not keras.layers.Dense:
                raise ValueError("Unsupported Layer Type: %s Only Dense Layers are Supported", type(layer))

            if l == 0:
                input_size = layer._input_shape_arg[0]
                for i in range(input_size):
                    graph.add_node(
                        self._unique_index(l, i),
                        shape="circle",
                        color="#3498db",
                        label=''
                    )
            output_size = layer.units

            for o in range(output_size):
                if l == len(self.model.layers) - 1:
                    graph.add_node(
                        self._unique_index(l + 1, o),
                        shape="circle",
                        color="#3498db",
                        label=''
                    )
                else:
                    graph.add_node(
                        self._unique_index(l + 1, o),
                        shape="circle",
                        color="#2ecc71",
                        label=''
                    )
                for i in range(input_size):
                    graph.add_edge(
                        self._unique_index(l, i),
                        self._unique_index(l + 1, o),
                        color="#B20000"
                    )
            input_size = layer.units
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
                int_model.add(self.model.layers[prev_layer])
            int_model.compile(loss=self.model.loss)
            intermediate_models.append(int_model)

        return intermediate_models


    def _snap_graph(self, filename):
        """
        Take snapshot image of the graph
        """
        out = to_agraph(self._graph)
        out.layout(prog='dot')
        out.draw(filename + '.png')
        return np.asarray(im.open(filename + '.png'))


    def _snap_X(self, indexes, X, Y, filename):
        """
        Take snapshot image of the input

        TODO:
            1. this doesn't work for X in > 2 dimension
            2. how to plot input could / should be specified by user

        Parameters:
            indexes: list of int.
                list of the indexes of the point in X that should be bold
            X: List.
                a list of coordinates of the points
            Y: List.
                a list of target values
            filename: str.
                name of file to which visualization will be saved
        """
        if X.shape[1] == 3:
            fig = plt.figure(figsize=(6, 6), frameon=False)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=COLORS[Y].tolist(), s=100, alpha=.9)
            for i in indexes:
                ax.scatter(X[i, 0], X[i, 1], X[i, 1], color=COLORS[Y[i]], s=300)
            fig.savefig(filename + '_X.png')
            plt.close()
        elif X.shape[1] == 2:
            fig, ax = plt.subplots(frameon=False)
            ax.scatter(X[:, 0], X[:, 1], color=COLORS[Y].tolist(), s=100, alpha=.9)
            for i in indexes:
                ax.scatter(X[i, 0], X[i, 1], color=COLORS[Y[i]], s=300, edgecolors='black')
            fig.savefig(filename + '_X.png')
            plt.close()
        return np.asarray(im.open(filename + '_X.png'))


    def _snap_decision_boundary(self, X, Y, filename, layer=0):
        """
        Take snapshot of input with decision boundary
        """

        # create a mesh to plot in
        # h = .02  # step size in the mesh
        # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                      np.arange(y_min, y_max, h))
        # meshData = np.c_[xx.ravel(), yy.ravel()]

        # if layer == 0:
        #     fig, ax = plt.subplots(frameon=False)
        #     ax.scatter(X[:, 0], X[:, 1], color=COLORS[Y].tolist(), s=100, alpha=.9)
        #     Z = self.model.predict(meshData).reshape(xx.shape)
        #     ax.contourf(xx, yy, Z, alpha=.4, cmap=CMAP)
        #     fig.savefig(filename + '.png')
        #     plt.close()

        #     return np.asarray(im.open(filename + '.png'))

        W = self.model.layers[layer].get_weights()[0]
        B = self.model.layers[layer].get_weights()[1]

        if X.shape[1] == 1:
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            fig, ax = plt.subplots(frameon=False)
            ax.scatter(X[:, 0], np.zeros_like(X[:, 0]), color=COLORS[Y].tolist(), s=100, alpha=.9)
            for i in range(W.shape[1]):
                xx = np.linspace(y_min, y_max, 200)
                ax.plot(np.zeros_like(xx) - B[i] / W[0][i], xx, linestyle='dashed', color='b')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        elif X.shape[1] == 2:
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            fig, ax = plt.subplots(frameon=False)
            ax.scatter(X[:, 0], X[:, 1], color=COLORS[Y].tolist(), s=100, alpha=.9)
            for i in range(W.shape[1]):
                xx = np.linspace(x_min, x_max, 200)
                ax.plot(xx, - W[0][i] * xx / W[1][i] - B[i] / W[1][i], linestyle='dashed', color='b')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        elif X.shape[1] == 3:
            h = .2
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            fig = plt.figure(figsize=(6, 6), frameon=False)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=COLORS[Y].tolist(), s=100, alpha=.9)
            for i in range(W.shape[1]):
                Z = - W[0][i] * xx / W[2][i] - W[1][i] * yy / W[2][i] - B[i] / W[2][i]
                ax.plot_surface(xx, yy, Z, color='y', alpha=.4)
        else:
            raise ValueError("Can't plot hidden layer with more > 3 dimensions")

        fig.savefig(filename + '.png')
        plt.close()

        return np.asarray(im.open(filename + '.png'))

    def _snap_regression(self, X, Y, filename):
        """
        Take snapshot of the regression line
        """
        x_min, x_max = X.min() - .5, X.max() + .5
        y_min, y_max = Y.min() - .5, Y.max() + .5
        xplot = np.linspace(x_min, x_max, 200)
        fig, ax = plt.subplots(frameon=False)
        ax.scatter(X, Y, color=COLORS[1], s=40, alpha=.7)
        ax.plot(xplot, self.model.predict(xplot.reshape(X.shape)), linewidth=2, color=COLORS[0])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        fig.savefig(filename + '.png')
        plt.close()
        return np.asarray(im.open(filename + '.png'))


    def _snap_feature_space(self, X, Y, filename):
        """
        Generate a snapshot of the feature space after transformation by the first hidden layer

        Parameters:
            X : ndarray
                input data to be transformed by the model's hidden layer
            Y : ndarray
                target classes corresponding to input data X, used for coloring the scatter plot
            filename : str
                name of file to save the snapshot as a PNG image

        Returns:
            np.ndarray
                Image array of the saved feature space snapshot
        """
        num_layers = len(self.model.layers)
        fig = plt.figure(figsize=(6 * num_layers, 6), frameon=False)

        for layer in range(num_layers):
            ax = fig.add_subplot(1, num_layers, layer + 1)
            if layer == 0:
                ax.imshow(self._snap_decision_boundary(X, Y, filename))
                ax.axis('off')
                ax.set_title("Input Space")
            else:
                feature_space = self._int_models[layer - 1].predict(X)
                ax.imshow(self._snap_decision_boundary(feature_space, Y, filename, layer))
                ax.axis('off')
                ax.set_title("Feature Space at Hidden Layer " + str(layer))

        fig.savefig(filename + '_combined.png', bbox_inches='tight')
        plt.close(fig)

        return np.asarray(im.open(filename + '_combined.png'))


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
            filename + '.gif',
            optimize=False,  # important for transparent background
            save_all=True,
            append_images=stacked_imgs[1:],
            loop=0,
            duration=duration
        )

        return


    def _convert_gif(self, imgs, filename, duration):
        """
        Convert a list of images to a gif.

        Args:
            imgs: List.
                a list of images
            filename: str.
                the filename for the gif
            duration: int
                duration in ms between images in GIF

        Returns:
            None

        """
        stacked_images = []
        for img in imgs:
            stacked_images.append(im.fromarray(np.asarray(img)))

        stacked_images[0].save(
            filename + '.gif',
            optimize=False,
            save_all=True,
            append_images=stacked_images[1:],
            loop=0,
            duration=duration
        )

        return


    def _reset(self):
        """
        Resets the graph labels, colors, fonts
        """
        self._graph = self._graph_original_copy.copy()

    def _normalize_activations(self, acts):
        """
        Normalize activations to [0, 1] for colormapping.

        This library often visualizes activations with colormaps that expect
        values in [0, 1], but activations can be outside that range depending
        on the layer (e.g., linear / tanh / relu pre/post).
        """
        a = np.asarray(acts, dtype=float)
        if a.size == 0:
            return a
        a_min = float(np.nanmin(a))
        a_max = float(np.nanmax(a))
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max == a_min:
            return np.zeros_like(a, dtype=float)
        return (a - a_min) / (a_max - a_min)

    def _apply_layer_node_styles(
        self,
        layer_index,
        acts,
        *,
        cmap,
        fill=True,
        show_sign_labels=False,
    ):
        """
        Apply node styles for a given layer index.

        Parameters:
            layer_index : int
                Graph layer index (0 = inputs, 1.. = subsequent layers).
            acts : array-like
                Activations for nodes in the specified layer.
            cmap : matplotlib colormap
                Colormap used to map normalized activations to colors.
            fill : bool
                Whether to use filled node styles.
            show_sign_labels : bool
                For input layer: optionally label '+' / '-' based on sign.
        """
        acts = np.asarray(acts).reshape(-1)
        normed = self._normalize_activations(acts)
        for n in range(len(acts)):
            index = self._unique_index(layer_index, n)
            if layer_index == 0 and show_sign_labels:
                set_node_attributes(self._graph, {index: {'label': "+" if acts[n] > 0 else "-"}})
            else:
                attrs = {
                    'style': 'filled' if fill else 'solid',
                    'color': str(rgb2hex(cmap(float(normed[n])))),
                }
                set_node_attributes(self._graph, {index: attrs})

    def view_forward_pass_for(
        self,
        x,
        filename="forward_pass",
        duration=600,
        steps_per_layer=1,
        show_input_signs=True,
        fade_previous=True,
    ):
        """
        Animate a single input's activations flowing through the network.

        Compared to `view_activations_for`, this renders a **layer-by-layer**
        animation: frame 0 shows the input, then each subsequent frame shows
        the next layer activating based on the same input.

        Parameters:
            x : array-like, shape (n_features,) or (1, n_features)
                A single input sample to feed forward through the network.
            filename : str
                Output file prefix. Saves `{filename}.gif` (and temporary PNGs).
            duration : int
                Duration in ms between GIF frames.
            steps_per_layer : int
                Number of interpolation steps between layers (>= 1).
                Higher values make the activation transition smoother.
            show_input_signs : bool
                If True, label input nodes with '+' / '-' based on sign.
            fade_previous : bool
                If True, previously-activated layers are kept visible but
                slightly muted to emphasize the current layer.
        """
        steps_per_layer = max(1, int(steps_per_layer))

        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.shape[0] != 1:
            raise ValueError("view_forward_pass_for expects a single input sample (shape (n_features,) or (1, n_features))")

        # Build activations for each graph-layer: inputs then each Dense output.
        activations = [x_arr[0]]
        for m in self._int_models:
            activations.append(np.asarray(m.predict(x_arr))[0])
        # `activations` length is (num_layers + 1), matching graph layer indices 0..L

        # Colormaps: green-ish for hidden layers, purple/blue for outputs.
        cvals = [0, 1]
        colors = ['white', 'green']
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        CMAP_GREEN = LinearSegmentedColormap.from_list("", tuples, 100)

        images = []

        # Pre-style input layer once (labels optional).
        self._apply_layer_node_styles(
            0,
            activations[0],
            cmap=CMAP_GREEN,
            fill=False,
            show_sign_labels=show_input_signs,
        )
        images.append(im.fromarray(self._snap_graph(filename)))

        # Animate layer-by-layer activation.
        last_layer = len(activations) - 1
        for layer_idx in range(1, last_layer + 1):
            acts = activations[layer_idx]
            is_output = (layer_idx == last_layer)
            cmap = CMAP if is_output else CMAP_GREEN

            # Interpolate from "off" to actual activations for smoother playback.
            off = np.zeros_like(np.asarray(acts, dtype=float))
            for s in range(1, steps_per_layer + 1):
                t = s / steps_per_layer
                blended = off * (1 - t) + np.asarray(acts, dtype=float) * t

                if fade_previous:
                    # Muted previous layers for context.
                    for prev in range(1, layer_idx):
                        prev_acts = activations[prev]
                        prev_is_output = (prev == last_layer)
                        prev_cmap = CMAP if prev_is_output else CMAP_GREEN
                        self._apply_layer_node_styles(prev, prev_acts, cmap=prev_cmap, fill=True)

                self._apply_layer_node_styles(layer_idx, blended, cmap=cmap, fill=True)
                images.append(im.fromarray(self._snap_graph(filename)))

        self._convert_gif(images, filename, duration)
        self._reset()
        return


    def fit(self, X, Y, snap_freq=10, filename='decision_boundary', duration=1000, view_feature_space=False, **kwargs):
        """
        Make GIF from snapshots of decision boundary at given snap_freq of epochs during training

        Parameters:
            X : ndarray
                input to a Keras model
            Y : ndarray
                classes to be learned
            snap_freq : int
                number of epochs after which to take a snapshot
            filename : str
                name of file to save as GIF
            duration : int
                duration in ms between images in GIF
            view_feature_space : bool
                flag to display the decision boundary in hidden feature space
            **kwargs : other params
                paramter inputs to model.fit

        Returns:
            The model after learning
        """

        images = []
        if 'epochs' in kwargs:
            epochs = kwargs['epochs']
            kwargs.pop('epochs', None)
        else:
            epochs = snap_freq

        for _ in range(int(epochs / snap_freq)):
            self.model.fit(X, Y, epochs=snap_freq, **kwargs)
            self._int_models = self._get_int_models()  # TODO: make this function more efficient

            if self.model.loss == "binary_crossentropy":
                if view_feature_space:
                    images.append(im.fromarray(self._snap_feature_space(X, Y, filename)))
                else:
                    images.append(im.fromarray(self._snap_decision_boundary(X, Y, filename)))
            elif self.model.loss == "mean_squared_error":
                images.append(im.fromarray(self._snap_regression(X, Y, filename)))
            else:
                raise ValueError("model loss not supported")

        self._convert_gif(images, filename, duration)
        return self.model


    def view_activations_for(self, X, Y, filename='activations', duration=1000):
        """
        Creates an animation of the graph activated by each data point

        Parameters:
            X : ndarray
                input to a Keras model
            filename : str
                name of file to which visualization will be saved
            duration : int
                duration in ms between images in GIF
            x_color: str.
                the color (in hex form) of the points in the pyplot graph
            x_marker: str.
                the shape of the points in the pyplot graph

        Returns:
            None
        """

        cvals = [0, 1]
        colors = ['white', 'green']
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        CMAP_GREEN = LinearSegmentedColormap.from_list("", tuples, 100)

        network_images = []
        input_images = []

        predictions = [X]
        for i in range(len(self._int_models)):
            predictions.append(self._int_models[i].predict(X))
        predictions.append(self.model.predict(X))

        for i in range(len(X)):
            input_size = self.model.layers[0]._input_shape_arg[0]
            for l in range(len(self.model.layers)):
                if l != 0:
                    input_size = self.model.layers[l - 1].units

                for n in range(0, input_size):
                    act = predictions[l][i][n]
                    index = self._unique_index(l, n)
                    if l == 0:
                        set_node_attributes(self._graph, {
                            index: {
                                'label': "+" if act > 0 else "-"
                            }})
                    else:
                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(CMAP_GREEN(act)))
                            }})

                for h in range(0, self.model.layers[l].units):
                    if l == len(self.model.layers) - 1:
                        act = predictions[l + 1][i][h]
                        index = self._unique_index(l + 1, h)

                        set_node_attributes(self._graph, {
                            index: {
                                'style': 'filled',
                                'color': str(rgb2hex(CMAP(act)))
                            }})

            network_images.append(self._snap_graph(filename))
            input_images.append(self._snap_X([i], X, Y, filename))
            self._reset()

        self._stack_gifs(network_images, input_images, filename, duration)
        return


    def render(self, filename='graph'):
        """
        Creates a visualization of the graph for a Sequential Dense keras model

        Parameters:
            filename : str
                name of file to which visualization will be saved

        Returns:
            None
        """

        self._snap_graph(filename)
        return


class NetworkArchitectureVisualizer:
    """
    Utility for rendering an arbitrary network architecture diagram using
    NetworkX + PyGraphviz (Graphviz).

    This is intentionally model-agnostic: provide nodes/edges (or an existing
    NetworkX graph) and render to an image using Graphviz layout.
    """

    def __init__(self, graph=None, directed=True):
        """
        Parameters:
            graph : networkx.Graph | networkx.DiGraph | None
                Existing graph to render. If None, an empty graph is created.
            directed : bool
                If graph is None, controls whether to create a DiGraph (True)
                or Graph (False).
        """
        if graph is None:
            self.graph = DiGraph() if directed else Graph()
        else:
            self.graph = graph

    @classmethod
    def from_keras_model(
        cls,
        model,
        *,
        weight_styling=False,
        input_color="#3498db",
        hidden_color="#2ecc71",
        output_color="#3498db",
        border_color="#2c3e50",
        node_border_width="0.8",
        skip_non_dense=True,
    ):
        """
        Build a NetworkArchitectureVisualizer directly from a Keras
        Sequential model whose trainable layers are Dense.

        Node ids follow the convention ``f"{layer_idx}_{neuron_idx}"`` where
        ``layer_idx == 0`` is the input layer.

        Parameters:
            model : keras.Model
                Sequential / functional model. Only Dense layers are walked.
            weight_styling : bool
                If True, edges are colored / sized by their corresponding
                weight magnitude and sign (zeros become faint dashed). This
                is the same as constructing the graph and then calling
                ``style_edges_by_weights(model)``.
            input_color, hidden_color, output_color : str
                Fill colors for input, hidden, and output neurons.
            border_color : str
                Outline color used for every neuron.
            node_border_width : str
                Graphviz ``penwidth`` for the node outline.
            skip_non_dense : bool
                If True, non-Dense layers (Dropout, BatchNorm, etc.) are
                silently skipped. If False, the method raises on the first
                non-Dense layer.

        Returns:
            NetworkArchitectureVisualizer
        """
        dense_layers = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                dense_layers.append(layer)
            elif not skip_non_dense:
                raise ValueError(
                    "from_keras_model only supports Dense layers; got "
                    f"{type(layer).__name__}. Pass skip_non_dense=True to ignore."
                )

        if not dense_layers:
            raise ValueError(
                "from_keras_model requires at least one Dense layer."
            )

        graph = DiGraph()
        in_dim = int(dense_layers[0].get_weights()[0].shape[0])

        for i in range(in_dim):
            graph.add_node(
                f"0_{i}",
                shape="circle",
                style="filled",
                fillcolor=input_color,
                color=border_color,
                penwidth=node_border_width,
                label="",
            )

        prev_dim = in_dim
        weight_matrices = []
        for layer_idx, layer in enumerate(dense_layers, start=1):
            W = layer.get_weights()[0]
            out_dim = int(W.shape[1])
            is_output = (layer_idx == len(dense_layers))
            node_color = output_color if is_output else hidden_color

            for o in range(out_dim):
                graph.add_node(
                    f"{layer_idx}_{o}",
                    shape="circle",
                    style="filled",
                    fillcolor=node_color,
                    color=border_color,
                    penwidth=node_border_width,
                    label="",
                )
                for i in range(prev_dim):
                    graph.add_edge(
                        f"{layer_idx - 1}_{i}",
                        f"{layer_idx}_{o}",
                    )

            weight_matrices.append(W)
            prev_dim = out_dim

        instance = cls(graph)
        if weight_styling:
            instance.style_edges_by_weights(weight_matrices)
        return instance

    def style_edges_by_weights(
        self,
        weights,
        *,
        pos_color="#B20000",
        neg_color="#1f3fa8",
        zero_color="#dddddd",
        min_width=0.35,
        max_width=3.0,
        zero_threshold=0.0,
    ):
        """
        Color and size edges based on signed weight magnitude.

        Edges with ``|w| <= zero_threshold`` are drawn faint and dashed,
        which is exactly the visual you want for showing pruning. Positive
        weights use ``pos_color``, negative weights use ``neg_color``, and
        each layer is normalized by its own max-abs weight so a layer of
        small weights still produces visible edges.

        Parameters:
            weights : keras.Model | Iterable[np.ndarray]
                Either a Keras model (Dense layers are extracted in order)
                or an iterable of 2D weight matrices in layer order. Edges
                between layer L-1 and layer L use ``weights[L-1]``.
            pos_color, neg_color, zero_color : str
                Edge colors for positive, negative, and effectively-zero
                weights.
            min_width, max_width : float
                Range mapped to ``penwidth`` based on normalized magnitude.
            zero_threshold : float
                Absolute magnitude at or below which a weight is treated as
                zero (drawn dashed).

        Returns:
            self
        """
        if hasattr(weights, "layers"):
            mats = []
            for layer in weights.layers:
                if isinstance(layer, keras.layers.Dense):
                    mats.append(layer.get_weights()[0])
            weights = mats

        weights = [np.asarray(W) for W in weights]
        if not weights:
            return self

        for layer_idx, W in enumerate(weights, start=1):
            denom = float(np.max(np.abs(W))) + 1e-12
            in_dim, out_dim = W.shape
            for i in range(in_dim):
                for o in range(out_dim):
                    u = f"{layer_idx - 1}_{i}"
                    v = f"{layer_idx}_{o}"
                    if not self.graph.has_edge(u, v):
                        continue
                    w = float(W[i, o])
                    if abs(w) <= zero_threshold:
                        self.graph.edges[u, v].update(
                            color=zero_color,
                            style="dashed",
                            penwidth=f"{min_width:.2f}",
                        )
                    else:
                        color = pos_color if w > 0 else neg_color
                        width = min_width + (max_width - min_width) * (
                            abs(w) / denom
                        )
                        self.graph.edges[u, v].update(
                            color=color,
                            style="solid",
                            penwidth=f"{width:.2f}",
                        )
        return self

    def add_nodes(self, nodes, default_attrs=None):
        """
        Add nodes to the graph.

        Parameters:
            nodes : Iterable[str] | Iterable[tuple[str, dict]]
                Either a list of node ids, or (node_id, attrs_dict) tuples.
            default_attrs : dict | None
                Attributes applied to nodes that are provided as ids only.
        """
        default_attrs = default_attrs or {}
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                node_id, attrs = n
                self.graph.add_node(node_id, **attrs)
            else:
                self.graph.add_node(n, **default_attrs)
        return self

    def add_edges(self, edges, default_attrs=None):
        """
        Add edges to the graph.

        Parameters:
            edges : Iterable[tuple[str, str]] | Iterable[tuple[str, str, dict]]
                Either (src, dst) tuples, or (src, dst, attrs_dict) tuples.
            default_attrs : dict | None
                Attributes applied to edges without explicit attrs.
        """
        default_attrs = default_attrs or {}
        for e in edges:
            if isinstance(e, tuple) and len(e) == 3 and isinstance(e[2], dict):
                u, v, attrs = e
                self.graph.add_edge(u, v, **attrs)
            else:
                u, v = e
                self.graph.add_edge(u, v, **default_attrs)
        return self

    def render(
        self,
        filename="network_architecture",
        layout="dot",
        graph_attrs=None,
        default_node_attrs=None,
        default_edge_attrs=None,
        theme="layered",
    ):
        """
        Render the architecture to a PNG using Graphviz.

        Parameters:
            filename : str
                Output file prefix ('.png' is appended).
            layout : str
                Graphviz layout program (e.g. 'dot', 'neato', 'fdp', 'sfdp').
            graph_attrs : dict | None
                Attributes applied to the overall graph (e.g. rankdir, splines).
                Merged on top of the theme; explicit values win.
            default_node_attrs : dict | None
                Default node styling applied via Graphviz (Agraph.node_attr).
                Merged on top of the theme; explicit values win.
            default_edge_attrs : dict | None
                Default edge styling applied via Graphviz (Agraph.edge_attr).
                Merged on top of the theme; explicit values win.
            theme : str | dict | None
                Built-in styling preset. One of "layered" (default),
                "weighted", "pruned", or a dict with the same shape as
                entries in ``NETWORK_ARCH_THEMES``. Pass ``theme=None`` for
                the previous bare-Graphviz behavior.
        """
        g_attrs, n_attrs, e_attrs = _resolve_theme(theme)
        if graph_attrs:
            g_attrs.update(graph_attrs)
        if default_node_attrs:
            n_attrs.update(default_node_attrs)
        if default_edge_attrs:
            e_attrs.update(default_edge_attrs)

        A = to_agraph(self.graph)

        for k, v in g_attrs.items():
            A.graph_attr[k] = v
        for k, v in n_attrs.items():
            A.node_attr[k] = v
        for k, v in e_attrs.items():
            A.edge_attr[k] = v

        A.layout(prog=layout)
        A.draw(filename + ".png")
        return filename + ".png"

    def _repr_png_(self):
        """
        Render to PNG bytes for inline Jupyter display.

        This makes ``NetworkArchitectureVisualizer.from_keras_model(model)``
        render automatically as the last expression in a notebook cell
        (no need to call ``render`` and ``display(Image(...))`` yourself).
        Uses the default ``theme="layered"`` styling.
        """
        if len(self.graph) == 0:
            return None
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = self.render(filename=os.path.join(td, "preview"))
            with open(path, "rb") as f:
                return f.read()
