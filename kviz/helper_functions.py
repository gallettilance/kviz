from matplotlib.colors import ListedColormap
import numpy as np
import re
import random


# According to https://graphviz.org/doc/info/shapes.html.
# There are three main types of shapes : polygon-based, record-based and user-defined.
# For now, this list only supports some polygon-based shapes.
valid_graphviz_shapes = [
    "box", "polygon", "ellipse", "oval", "circle", "egg", "triangle", "diamond", "trapezium",
    "parallelogram", "house", "pentagon", "hexagon", "septagon", "octagon", "doublecircle", "doubleoctagon",
    "tripleoctagon", "invtriangle", "invtrapezium", "invhouse", "Mdiamond", "Msquare", "Mcircle", "rect", "rectangle",
    "square", "star", "cylinder",
]

chars = '0123456789ABCDEF'


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


def check_regular_expression_for_color(color):
    """
        Check if the color string is a valid hex form (e.g. "#FFFFFF").
        True if the string is valid; False otherwise.

        Parameters:
            color: str.
                The color string to check.

        Returns: bool.

    """
    return re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)


def check_valid_networkx_shape(shape):
    """
        Check if the shape string is a valid for networkx. (e.g. "polygon").
        Check https://graphviz.org/doc/info/shapes.html for some valid shapes.
        For now, most polygon-based shapes are supported.

        Parameters:
            shape: str.
                The shape.

        Returns: bool.

    """
    return shape in valid_graphviz_shapes


def get_random_shape():
    """
        Randomly returns a shape.

        Returns: str.

    """
    return random.choice(valid_graphviz_shapes)


def get_random_color():
    """
        Randomly returns a hex color string.

        Returns: str.

    """
    return "#" + "".join([random.choice(chars) for _ in range(6)])
