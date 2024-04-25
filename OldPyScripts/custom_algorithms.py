import math
from math import sqrt
from OldPyScripts.intersection_area_utils import calc_intersect_area, find_extreme_vertices


def ellipse_diff_weight_distance(point1, point2, width, height, slope, xdt, ydt):
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    if slope == 0:
        slope = 0.0001
    theta = math.atan(-1 / slope)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    rotated_x_diff = x_diff * cos_theta + y_diff * sin_theta
    rotated_y_diff = y_diff * cos_theta - x_diff * sin_theta

    perpen_diff = rotated_x_diff / (width * xdt)
    parall_diff = rotated_y_diff / (height * ydt)

    dist = sqrt(perpen_diff ** 2 + parall_diff ** 2)
    # if rotated_y_diff > 0:
    #     return -dist
    return dist


def ellipse_same_weight_distance(point1, point2, width, height, slope, dt):
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    if slope == 0:
        slope = 0.0001
    theta = math.atan(-1 / slope)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    rotated_x_diff = x_diff * cos_theta + y_diff * sin_theta
    rotated_y_diff = y_diff * cos_theta - x_diff * sin_theta

    perpen_diff = rotated_x_diff / (width * dt)
    parall_diff = rotated_y_diff / (height * dt)

    dist = math.sqrt(perpen_diff ** 2 + parall_diff ** 2)
    # if rotated_y_diff > 0:
    #     return -dist
    return dist


def vertical_line_distance(point2, vertice1, vertice2, slope):
    if slope == 0:
        slope = 0.0001
    line1_c = vertice1[0] - slope * vertice1[1]  # calculate line1 c in y = slope*x + c
    line2_c = vertice2[0] - slope * vertice2[1]  # calculate line2 c in y = slope*x + c
    line1_x = (point2[1] - line1_c) / slope      # calculate x when point2's y in line1 function
    line2_x = (point2[1] - line2_c) / slope      # calculate x when point2's y in line2 function
    w = line2_x - line1_x                        # calculate x distance between line1 and line2
    a = (point2[0] - line1_x) / w                # calculate x distance ratio between point2 and line1
    b = (line2_x - point2[0]) / w                # calculate x distance ratio between point2 and line2
    dist = 1 - 4 * a * b

    return dist


def collapse_area(t1_vertices, t2_vertices, slope, for_plot=False):
    # if slope == 0:
    #     slope = 0.0001
    slope = 200
    left_top, left_bottom, right_bottom, right_top = find_extreme_vertices(t1_vertices)
    line1_point = (left_top[0], -left_top[1])
    line2_point = (right_top[0], -right_top[1])
    if for_plot:
        return (-slope, line1_point), (-slope, line2_point)
    else:
        area, collapse_vertices = calc_intersect_area(t2_vertices, (-slope, -slope), (line1_point, line2_point))
        return area
