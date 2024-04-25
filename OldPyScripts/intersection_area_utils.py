import math


def centroid(vertices):
    """
    Calculate the centroid of a polygon.
    :param vertices: List of (x, y) tuples representing the polygon's vertices.
    :return: Tuple (Cx, Cy) representing the coordinates of the centroid.
    """
    x, y = zip(*vertices)
    return sum(x) / len(vertices), sum(y) / len(vertices)


def sort_vertices_clockwise(vertices):
    """
   Sorts the vertices of a polygon in a clockwise order based on their angle with the centroid.
   :param vertices: List of (x, y) tuples representing the polygon's vertices.
   :return: List of vertices sorted in clockwise order.
   """
    Cx, Cy = centroid(vertices)
    vertices.sort(key=lambda v: -math.atan2(v[1] - Cy, v[0] - Cx))
    return vertices


def shoelace_area(vertices):
    """
    Calculates the area of a polygon using the shoelace theorem.
    :param vertices: List of (x, y) tuples in clockwise order.
    :return: The area of the polygon.
    """
    return abs(
        sum(
            x0 * y1 - y0 * x1 for (x0, y0), (x1, y1)
            in zip(vertices, vertices[1:] + vertices[:1])
        )
    ) / 2


def line_intersection(m, c, segment):
    """
    Finds the intersection point of a line and a line segment if it exists.
    :param m: Slope of the line.
    :param c: Y-intercept of the line.
    :param segment: Tuple of points (x1, y1), (x2, y2) representing the line segment.
    :return: The intersection point (x, y) or None if no intersection exists.
    """
    (x1, y1), (x2, y2) = segment

    # Segment's slope and intercept
    if x2 != x1:
        m2 = (y2 - y1) / (x2 - x1)
        c2 = y1 - m2 * x1
    else:
        # Check if the main line is also vertical with the same x
        if m == float('inf') and x1 == -c:
            # Both are vertical and overlapping
            return segment
        else:
            # Vertical segment not parallel to line
            y = m * x1 + c
            if min(y1, y2) <= y <= max(y1, y2):
                return (x1, y)
            return None

    # Check if the lines are parallel
    if m == m2:
        if c == c2:  # Check if they overlap
            # They are parallel and on the same line
            if (min(x1, x2) <= -c/m <= max(x1, x2) or
                    min(y1, y2) <= (m * min(x1, x2) + c) <= max(y1, y2)):
                return segment
        return None

    # Solve for intersection
    if m != m2:  # Ensure lines are not parallel
        x = (c2 - c) / (m - m2)
        y = m * x + c

        # Check if the intersection point is within the segment bounds
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return x, y
    return None
    # if x1 == x2:  # Segment is vertical
    #     y = -line_value(x1, 0, line_slope, line_intercept)
    #     return (x1, y) if min(y1, y2) <= y <= max(y1, y2) else None
    # segment_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    # if segment_slope == line_slope:  # Parallel lines
    #     return None
    # x = (-line_value(0, 0, -line_slope, line_intercept) - segment_slope * x1 + y1) / (line_slope - segment_slope)
    # y = -line_value(x1, 0, line_slope, line_intercept)
    # return (x, y) if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) else None


def polygon_line_intersections(vertices, line_slope, line_intercept):
    """
    Finds all intersection points between a polygon and a line.
    :param vertices: List of the polygon's vertices.
    :param line_slope: Slope of the line.
    :param line_intercept: Y-intercept of the line.
    :return: List of intersection points (x, y).
    """
    results = []
    intercept = line_value(0, 0, -line_slope, line_intercept)
    # print()
    # print(f"line slope: {line_slope}, line_intercept: {intercept}")
    for i in range(len(vertices)):
        edge = (vertices[i], vertices[(i + 1) % len(vertices)])
        intersection = line_intersection(line_slope, intercept, edge)
        # print(f"Checking edge from {edge[0]} to {edge[1]}: Intersection found at {intersection}")
        if intersection:
            results.append(intersection)
    # print()
    return results
    # return [p for p in
    #         (
    #             line_intersection(
    #                 line_slope,
    #                 line_intercept,
    #                 (vertices[i], vertices[(i + 1) % len(vertices)])
    #             ) for i in range(len(vertices))
    #         ) if p]


def line_value(x, y, slope, intercept):
    """
    Computes the value of the line function at point (x, y).
    :param x: X-coordinate of the point.
    :param y: Y-coordinate of the point.
    :param slope: Slope of the line.
    :param intercept: A tuple (x1, y1) representing a point through which the line passes.
    :return: The value of the line equation at the point.
    """
    x1, y1 = intercept
    return y - (slope * (x - x1) + y1)


def find_include_vertices(vertices, slope, intercept, line_side='left'):
    """
    Finds vertices on the specified side of a line.
    :param vertices: List of vertices to check.
    :param slope: Slope of the line.
    :param intercept: Y-intercept of the line.
    :param line_side: 'left' for right side of left line, 'right' for left side of right line.
    :return: List of vertices on the specified side of the line.
    """
    sign = slope / abs(slope)
    check = lambda v: \
        (line_value(*v, slope, intercept) * sign <= 0) \
            if line_side == 'left' \
            else line_value(*v, slope, intercept) * sign >= 0
    return [v for v in vertices if check(v)]


def order_lines_by_position(slopes, points):
    """
    Orders two lines based on their position, with the line on the left first.

    Considers negative and positive slopes:
    - For negative slopes: Higher intercept indicates the line is to the right.
    - For positive slopes: Lower intercept indicates the line is to the right.

    :param slopes: tuple of slope of lines.
    :param points: tuple of points (x1, y1) through which each line passes.
    :return: Tuple of (left line, right line), each line as (slope, point).
    """
    # Determine which line is to the right based on the slope sign and intercepts
    x_at_y0_line1 = (-points[0][1] / slopes[0]) + points[0][0] if slopes[0] != 0 else float('inf')
    x_at_y0_line2 = (-points[1][1] / slopes[1]) + points[1][0] if slopes[1] != 0 else float('inf')
    line1_c = x_at_y0_line1 * -slopes[0]
    line2_c = x_at_y0_line2 * -slopes[1]
    # print(f"x_at_y0_line1: {x_at_y0_line1}")
    # print(f"x_at_y0_line2: {x_at_y0_line2}")

    # Determine which line is to the left based on the x-coordinates at y = 0
    if x_at_y0_line1 < x_at_y0_line2:
        return (slopes[0], points[0]), (slopes[1], points[1])
    else:
        return (slopes[1], points[1]), (slopes[0], points[0])


def find_extreme_vertices(vertices):
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


def calc_intersect_area(vertices, slopes, intercepts):
    """
    Calculates the area of the polygon formed by intersections and vertices inclusion between two lines.
    :param vertices: List of the polygon's vertices.
    :param slopes: List of two slopes for the two lines.
    :param intercepts: List of two intercepts for the two lines.
    :return: Tuple of area and the vertices that form the calculated area.
    """
    extreme_vertices = find_extreme_vertices(vertices)
    ((slope1, intersect1),
     (slope2, intersect2)) = order_lines_by_position(slopes, intercepts)
    # print(f"vertices: {extreme_vertices}")
    # print(f"lines: {((slope1, intersect1), (slope2, intersect2))}")
    include_left = find_include_vertices(extreme_vertices, slope1, intersect1, 'left')
    include_right = find_include_vertices(extreme_vertices, slope2, intersect2, 'right')
    # print(f"Left: {include_left}")
    # print(f"Right: {include_right}")

    if not include_left or not include_right:
        return 0, []
    # print(f"Left Intercept: {polygon_line_intersections(extreme_vertices, slope1, intersect1)}")
    # print(f"Right Intercept: {polygon_line_intersections(extreme_vertices, slope2, intersect2)}")
    intersections = (
            polygon_line_intersections(extreme_vertices, slope1, intersect1) +
            polygon_line_intersections(extreme_vertices, slope2, intersect2)
    )
    include = find_include_vertices(include_right, slope1, intersect1, 'left')
    # print(f"include: {include}")
    intersected_vertices = include + intersections
    if len(intersected_vertices) == 0:
        sorted_vertices = []
    else:
        sorted_vertices = sort_vertices_clockwise(list(set(intersected_vertices)))  # Remove duplicates and sort

    return shoelace_area(sorted_vertices), sorted_vertices


# def calc_intersect_area(vertices, metric_func):
#     """
#     Calculates the area of the polygon formed by intersections and vertices inclusion within the metric function.
#     :param vertices: List of the polygon's vertices.
#     :param metric_func: Function of metric of the area .
#     :return: Tuple of area and the vertices that form the calculated area.
#     """
#     extreme_vertices = find_extreme_vertices(vertices)
