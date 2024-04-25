from enum import Enum
from pandas.core.series import Series
import re


class Criteria(Enum):
    def __init__(self, type: str, equal_dims: bool, description: str):
        """
        :param type: Circular, Linear or Others
        :param equal_dims: whether to use same weight for x and y
        :param description:
        """
        self.type = type
        self.equal_dims = equal_dims
        self.description = description

    def describe(self):
        return f"This method is using '{self.description}' to evaluate text relationships."


class DistanceCriteria(Criteria):
    ESWD = ('Circular', True, 'ellipse same weight distance')
    EDWD = ('Circular', False, 'ellipse different weight distance')
    VTLD = ('Linear', None, 'vertical two lines distance')
    OTHER = ('Others', None, 'others')


class AreaCriteria(Criteria):
    ESWA = ('Circular', True, 'ellipse same weight area')
    EDWA = ('Circular', False, 'ellipse different weight area')
    VTLA = ('Linear', None, 'vertical two lines area')
    OTHER = ('Others', None, 'others')


class ImportVariables:
    def __init__(self, xdt: float, ydt: float, metric: Criteria):
        """
        :param xdt: weight of x
        :param ydt: weight of y
        :param metric: metric selected
        :return:
        """
        self.metric = metric
        self.xdt = xdt
        self.ydt = ydt
        self.img_shape = None

    def get_variables(self):
        return self.xdt, self.ydt, self.img_shape, self.metric

    def set_img_shape(self, new_shape: tuple):
        self.img_shape = new_shape


def normalize(text) -> str:
    """
    Normalizes the input text by converting it to lowercase and removing all non-alphanumeric
    characters except for spaces.

    :param text: The string to be normalized. If not a string, it will be converted to string.
    :return: A normalized string with all characters in lowercase, excluding any characters
             that are not letters, digits, or spaces.
    """
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())


def ensure_string(value) -> str:
    if isinstance(value, str):
        return value
    else:
        return str(value)


def flip_y(y: int, min_y: int, img_height: int) -> int:
    """
    Flip the y-coordinate in an image's coordinate system.
    :param y: int -- The original y-coordinate to be flipped.
    :param min_y: int -- The minimum y-value in the coordinate system.
    :param img_height: int -- The total height of the image.
    :return: The new y-coordinate after flipping.
    """
    return img_height - (y - min_y) + min_y


def find_extreme_vertices(vertices: list) -> (tuple, tuple, tuple, tuple):
    """
    Finds the extreme vertices of a polygon based on x and y coordinates.
    :param vertices: List of vertices of the polygon.
    :return: Tuple of (left_top, left_bottom, right_top, right_bottom) vertices.
    """

    # Sorting by x coordinate
    if not vertices or len(vertices) < 4:
        return None  # Ensure there are enough vertices to form a polygon
    vertices = [(x, -y) for (x, y) in vertices]

    x_sorted_vertices = sorted(vertices, key=lambda v: v[0])

    left_vertices = x_sorted_vertices[:2]
    right_vertices = x_sorted_vertices[-2:]

    y_sorted_left_vertices = sorted(left_vertices, key=lambda v: v[1])
    left_top, left_bottom = y_sorted_left_vertices[1], y_sorted_left_vertices[0]
    y_sorted_right_vertices = sorted(right_vertices, key=lambda v: v[1])
    right_top, right_bottom = y_sorted_right_vertices[1], y_sorted_right_vertices[0]

    return left_top, left_bottom, right_bottom, right_top


def select_slope(slope: tuple, direction: str) -> float:
    """
    calculate target slope from 4
    :param slope: tuple -- (slope1, slope2, slope3, slope4)
    :param direction: str -- word direction vertical or horizontal
    :return: target slope
    """
    h_slope = (slope[0] + slope[2]) / 2
    v_slope = (slope[1] + slope[3]) / 2
    h_slope = float(1000) if h_slope == 0 else -1/h_slope
    target_slope = min(abs(h_slope), abs(v_slope))
    if "vertical" == direction:
        target_slope = -1 / target_slope if target_slope != 0 else 1000
    elif "horizontal" == direction:
        pass
    else:
        print("===========================")
        print("l1 Invalid Direction")
        print("===========================")
        return Exception("Invalid Direction")
    return target_slope


def calc_xy_len(point: Series, xdt: float, ydt: float) -> (float, float):
    """
    calculate x and y value with given weights
    :param point: Series -- a row of df
    :param xdt: float -- x weight
    :param ydt: float -- y weight
    :return:
        point_x_len: float -- x value
        point_y_len: float -- y value
    """
    if point.direction == 'vertical':
        point_x_len = point.font * xdt
        point_y_len = point.word_len * ydt
    elif point.direction == 'horizontal':
        point_y_len = point.font * ydt
        point_x_len = point.word_len * xdt
    else:
        print("===========================")
        print(f"{point.txt} Invalid Direction")
        print("===========================")
        return Exception("Invalid Direction")

    return point_x_len, point_y_len
