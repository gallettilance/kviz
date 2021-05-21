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
from tensorflow.keras import models


class ConvGraph():
    """
        Class for creating and rendering visualization of Keras
        Sequential Model with Convolutional Layers

        Attributes:
            model : tf.keras.Model
                a compiled keras sequential model

        Methods:
            render :
                Shows all the convolution activations

    """

    def __init__(self, model):
        self.model = model


    def _snap_layer(self, display_grid, scale, filename, xticks=None, yticks=None):
        fig, ax = plt.subplots(figsize=(int(scale * display_grid.shape[1]), int(scale * display_grid.shape[0])))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(True)
        ax.axis('off')
        ax.imshow(display_grid, aspect='auto')
        fig.savefig(filename + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        return np.asarray(im.open(filename + '.png'))


    def animate(self, X=None, filename='conv_animation'):
        """
        Render animation of a Convolutional layers based on a stream
        of input.

        Parameters:
            X : ndarray
                input to a Keras model - ideally of the same class
            filename : str
                name of file to which visualization will be saved

        Returns:
            None
        """

        layer_outputs = [layer.output for layer in self.model.layers]
        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        images_per_row = 8

        for i in range(len(self.model.layers)):
            # Ignore non-conv2d layers
            layer_name = self.model.layers[i].name
            if not layer_name.startswith("conv2d"):
                continue

            images = []
            heat = []
            for j in range(len(X)):
                activations = activation_model.predict(X[j])
                # Number of features in the feature map
                n_features = activations[i].shape[-1]
                # The feature map has shape (1, size, size, n_features).
                size = activations[i].shape[1]
                # Tiles the activation channels in this matrix
                n_cols = n_features // images_per_row
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                # Tiles each filter into a big horizontal grid
                for col in range(n_cols):
                    for row in range(images_per_row):
                        # Displays the grid
                        display_grid[
                            col * size: (col + 1) * size,
                            row * size: (row + 1) * size] = activations[i][0, :, :, col * images_per_row + row]

                snapped = self._snap_layer(
                    display_grid, 1.0 / size,
                    filename + "_" + layer_name,
                    xticks=np.linspace(-1, display_grid.shape[1], images_per_row + 1),
                    yticks=np.linspace(-1, display_grid.shape[0], n_cols + 1))
                heat.append(snapped)
                images.append(im.fromarray(snapped))

            images[0].save(
                filename + "_" + layer_name + '.gif',
                optimize=False,  # important for transparent background
                save_all=True,
                append_images=images[1:],
                loop=0,
                duration=100,
                transparency=255,  # prevent PIL from making background black
                disposal=2
            )

            heatmap = heat[0]
            for i in range(1, len(heat)):
                heatmap = np.where(heatmap < heat[i], heat[i], heatmap)
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(heatmap)
            fig.savefig(filename + "_" + layer_name + '_heatmap.png', transparent=True, bbox_inches='tight', pad_inches=0)

        return


    def render(self, X=None, filename='conv_filters'):
        """
        Render visualization of a Convolutional keras model

        Parameters:
            X : ndarray
                input to a Keras model
            filename : str
                name of file to which visualization will be saved

        Returns:
            None
        """

        layer_outputs = [layer.output for layer in self.model.layers]
        # Creates a model that will return these outputs, given the model input
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        images_per_row = 8

        for j in range(len(X)):
            activations = activation_model.predict(X[j])

            for i in range(len(activations)):
                # Ignore non-conv2d layers
                layer_name = self.model.layers[i].name
                if not layer_name.startswith("conv2d"):
                    continue

                # Number of features in the feature map
                n_features = activations[i].shape[-1]
                # The feature map has shape (1, size, size, n_features).
                size = activations[i].shape[1]
                # Tiles the activation channels in this matrix
                n_cols = n_features // images_per_row
                display_grid = np.zeros((size * n_cols, images_per_row * size))
                # Tiles each filter into a big horizontal grid
                for col in range(n_cols):
                    for row in range(images_per_row):
                        # Displays the grid
                        display_grid[
                            col * size: (col + 1) * size,
                            row * size: (row + 1) * size] = activations[i][0, :, :, col * images_per_row + row]

                self._snap_layer(
                    display_grid, 1. / size,
                    filename + "_" + str(j) + "_" + layer_name,
                    xticks=np.linspace(-1, display_grid.shape[1], images_per_row + 1),
                    yticks=np.linspace(-1, display_grid.shape[0], n_cols + 1))

        return
