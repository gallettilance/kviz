from matplotlib.colors import ListedColormap
import numpy as np


def create_colormap(hex_color_string, N=10, step=51):
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
    return ListedColormap(vals)


def get_or_create_colormap_with_dict(color, dictionary):
    """
        Use the color as the key, return the colormap from the dictionary.
        If the color does not exist in the dictionary, it will be added and a corresponding colormap
        will be created.

        Parameters:
            color: str.
                Should be a hex color string.
            dictionary: dict.
                The dictionary which uses hex color as the key and colormap as the value.

        Returns: matplotlib.colors.ListedColormap.

    """
    if color in dictionary:
        the_color_map = dictionary[color]
    else:
        the_color_map = create_colormap(color)
        dictionary[color] = the_color_map
    return the_color_map


def unique_index(layer, node):
    """
        Returns a unique index given the layer and the node.

        Parameters:
            layer: int.
                The index of the layer (starts from 0).
            node: int.
                The index of the node (starts from 0).

        Returns: str.
    """
    return str(layer) + "_" + str(node)
