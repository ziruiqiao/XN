from abc import ABC, abstractmethod
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt

from ActivePyTools.utils import flip_y

try:
    import cupy as cp
    xp = cp
    USE_CUDA = True
except ImportError:
    xp = np
    USE_CUDA = False


class ShapeIntersection(ABC):
    @abstractmethod
    def intersection_area(self, vertices) -> float:
        """
        Calculate the area of intersection with a given polygon defined by vertices.
        :param vertices: Vertices of the polygon.
        Returns:
        float: The area of the intersection.
        """
        pass

    @abstractmethod
    def draw(self, vertices, padding=10):
        """
        Draw the shape and the polygon, including highlighting their intersection area.
        :param vertices: Vertices of the polygon.
        :param padding: padding area around polygon
        """
        pass

    @abstractmethod
    def draw_on_image(self, vertices, image, padding=10):
        """
        Draw the shape and the polygon, including highlighting their intersection area on an image
        :param vertices: Vertices of the polygon.
        :param image:
        :param padding: padding area around polygon
        """
        pass


class CircularIntersection(ShapeIntersection):
    def __init__(self, center: tuple, x_len: float, y_len: float, slope: float):
        """
        :param center: (x, y) center of ellipse
        :param x_len: x axe length
        :param y_len: y axe length
        :param slope: slope of the ellipse
        """
        self.center = center
        self.x_len = x_len
        self.y_len = y_len
        if slope == 0:
            slope = 0.0001
        self.angle = math.atan(-1 / slope)
        self.ellipse = self.create_ellipse_cuda()

    def create_ellipse(self) -> Polygon:
        angle_rad = np.radians(self.angle)
        ellipse_points = []
        for t in np.linspace(0, 360, num=180):
            rad = np.radians(t)
            x = self.x_len * np.cos(rad)
            y = self.y_len * np.sin(rad)
            x_rotated = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            y_rotated = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            ellipse_points.append((x_rotated + self.center[0], y_rotated + self.center[1]))
        return Polygon(ellipse_points)

    def intersection_area(self, vertices: list) -> float:
        polygon = Polygon(vertices)
        intersection = self.ellipse.intersection(polygon)
        return intersection.area

    def create_ellipse_cuda(self) -> Polygon:
        angle_rad = xp.radians(self.angle)
        t_values = xp.linspace(0, 360, num=180)  # Degree values for the ellipse points
        rad_values = xp.radians(t_values)  # Convert degrees to radians

        x = self.x_len * xp.cos(rad_values)
        y = self.y_len * xp.sin(rad_values)

        # Rotate points
        x_rotated = x * xp.cos(angle_rad) + y * xp.sin(angle_rad)
        y_rotated = x * xp.sin(angle_rad) + y * xp.cos(angle_rad)

        # Translate points to the center
        x_final = x_rotated + self.center[0]
        y_final = y_rotated + self.center[1]

        # Convert CuPy arrays back to NumPy arrays for Shapely, if using CuPy
        if USE_CUDA:
            ellipse_points = list(zip(cp.asnumpy(x_final), cp.asnumpy(y_final)))
        else:
            ellipse_points = list(zip(x_final, y_final))
        return Polygon(ellipse_points)

    def draw(self, vertices: list, padding: int = 10):
        polygon = Polygon(vertices)
        intersection = self.ellipse.intersection(polygon)
        fig, ax = plt.subplots()
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, color='red', label='Polygon')
        x, y = self.ellipse.exterior.xy
        ax.fill(x, y, alpha=0.5, color='blue', label='Ellipse')
        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            ax.fill(x, y, alpha=0.5, color='green', label='Intersection Area')

        all_x = x + self.ellipse.exterior.xy[0]
        all_y = y + self.ellipse.exterior.xy[1]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        ax.invert_yaxis()
        ax.legend()
        plt.show()

    def draw_on_image(self, vertices: list, image: np.ndarray, padding: int = 10):
        polygon = Polygon(vertices)
        intersection = self.ellipse.intersection(polygon)

        x_coords, y_coords = [], []

        for line in [polygon.exterior, self.ellipse.exterior]:
            x, y = line.xy
            x_coords.extend(x)
            y_coords.extend(y)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            x_coords.extend(x)
            y_coords.extend(y)

        min_x = max(int(min(x_coords) - padding), 0)
        max_x = min(int(max(x_coords) + padding), image.shape[1])
        min_y = max(int(min(y_coords) - padding), 0)
        max_y = min(int(max(y_coords) + padding), image.shape[0])

        cropped_image = image[min_y:max_y, min_x:max_x]
        img_height = cropped_image.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(cropped_image, extent=[min_x, max_x, min_y, max_y])
        for line, color, label in zip(
                [polygon.exterior, self.ellipse.exterior],
                ['red', 'blue'],
                ['Polygon', 'Ellipse']
        ):
            x, y = line.xy
            ax.fill(x, [flip_y(yi, min_y, img_height) for yi in y], alpha=0.5, label=label, color=color)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            ax.fill(x, [flip_y(yi, min_y, img_height) for yi in y], alpha=0.5, color='green', label='Intersection Area')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.legend()
        plt.show()


class LinearIntersection(ShapeIntersection):
    def __init__(self, line1: tuple, line2: tuple):
        """
        :param line1: line1 start point and end point
        :param line2: line2 start point and end point
        """
        self.line1 = LineString([line1[0], line1[1]])
        self.line2 = LineString([line2[0], line2[1]])

    @classmethod
    def from_slopes_and_intercepts(cls, points: tuple, slopes: tuple, image_dimensions: tuple):
        """
        Alternative constructor that creates lines from slopes and intercepts.
        :param slopes: Slopes of the lines (slope1, slope2).
        :param points: points on the lines ((x1, y1), (x2, y2)).
        :param image_dimensions: Dimensions of the image (height, width).
        """
        # Define lines based on x_range
        height, width, _ = image_dimensions

        # Calculate line endpoints based on slope and a point, spanning the image width
        def calculate_line_endpoints(slope, point, img_width):
            x0, y0 = point
            # Find y-intercept using y = mx + b => b = y - mx
            y_at_x0 = - slope * x0 + y0
            y_at_xmax = slope * img_width + y_at_x0
            # print((0, y_at_x0), (width, y_at_xmax))

            return (0, y_at_x0), (img_width, y_at_xmax)

        line1 = calculate_line_endpoints(slopes[0], points[0], width)
        line2 = calculate_line_endpoints(slopes[1], points[1], width)

        # Instantiate the class with the calculated line endpoints
        return cls(line1, line2)

    def intersection_area(self, vertices: list) -> float:
        line_polygon = Polygon(list(self.line1.coords) + list(reversed(self.line2.coords)))
        polygon = Polygon(vertices)
        intersection = polygon.intersection(line_polygon)
        return intersection.area

    def draw(self, vertices: list, padding: int = 10):
        polygon = Polygon(vertices)
        line_polygon = Polygon(list(self.line1.coords) + list(reversed(self.line2.coords)))
        intersection = polygon.intersection(line_polygon)

        x_coords, y_coords = [], []

        x, y = polygon.exterior.xy
        x_coords.extend(x)
        y_coords.extend(y)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            x_coords.extend(x)
            y_coords.extend(y)

        min_x, max_x = min(x_coords) - padding, max(x_coords) + padding
        min_y, max_y = min(y_coords) - padding, max(y_coords) + padding

        fig, ax = plt.subplots()

        for line, color, label in zip(
                [self.line1, self.line2, polygon.exterior],
                ['blue', 'green', 'black'],
                ['Line 1', 'Line 2', 'Polygon']
        ):
            x, y = line.xy
            ax.plot(x, y, label=label, color=color, linewidth=2)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            ax.fill(x, y, alpha=0.5, color='red', label='Intersection Area')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.invert_yaxis()
        ax.legend()
        plt.show()

    def draw_on_image(self, vertices: list, image: np.ndarray, padding: int = 10):
        polygon = Polygon(vertices)
        line_polygon = Polygon(list(self.line1.coords) + list(reversed(self.line2.coords)))
        intersection = polygon.intersection(line_polygon)

        x_coords, y_coords = [], []

        x, y = polygon.exterior.xy
        x_coords.extend(x)
        y_coords.extend(y)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            x_coords.extend(x)
            y_coords.extend(y)

        min_x = max(int(min(x_coords) - padding), 0)
        max_x = min(int(max(x_coords) + padding), image.shape[1])
        min_y = max(int(min(y_coords) - padding), 0)
        max_y = min(int(max(y_coords) + padding), image.shape[0])

        # print(min_x, max_x, min_y, max_y)

        cropped_image = image[min_y:max_y, min_x:max_x]
        img_height = cropped_image.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(cropped_image, extent=[min_x, max_x, min_y, max_y])

        for line, color, label in zip(
                [self.line1, self.line2, polygon.exterior],
                ['blue', 'green', 'black'],
                ['Line 1', 'Line 2', 'Polygon']
        ):
            x, y = line.xy
            ax.plot(x, [flip_y(yi, min_y, img_height) for yi in y], label=label, color=color, linewidth=2)

        if not intersection.is_empty:
            x, y = intersection.exterior.xy
            ax.fill(x, [flip_y(yi, min_y, img_height) for yi in y], alpha=0.5, color='red', label='Intersection Area')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.legend()
        plt.show()


class PointDistance(ABC):
    @abstractmethod
    def calc_dist(self, center_point) -> float:
        """
        Calculate the distance between the center_point and the metrics.
        :param center_point: center point of a word.
        :return: distance between the center_point and the metrics
        """
        pass

    @abstractmethod
    def draw(self, center_point):
        """
        Draw the shape and the center_point,
        :param center_point: center point of a word.
        """
        pass

    @abstractmethod
    def draw_on_image(self, center_point, image):
        """
        Draw the shape and the center_point on an image
        :param center_point: center point of a word.
        :param image:
        """
        pass


class LinearDist(PointDistance):
    def __init__(self, points: tuple, slopes: tuple):
        non_zero_slopes = cp.array([slope if slope != 0 else 0.0001 for slope in slopes])

        self.points = cp.array([(x, y) for (x, y) in points])
        self.slopes = non_zero_slopes

        self.line1_c = points[0][1] - self.slopes[0] * points[0][0]
        self.line2_c = points[1][1] - self.slopes[1] * points[1][0]

    def calc_dist(self, center_point: tuple) -> float:
        center_x, center_y = center_point[0], center_point[1]
        # Calculate x for given y
        line1_x = (center_y - self.line1_c) / self.slopes[0]
        line2_x = (center_y - self.line2_c) / self.slopes[1]
        # Width and ratios
        w = line2_x - line1_x
        a = (center_x - line1_x) / w
        b = (line2_x - center_x) / w
        dist = 1 - 4 * a * b
        return cp.asnumpy(dist).item()

    def calculate_line_values(self, x_range: tuple):
        # Calculate y-values for the two lines
        x_vals = cp.array([x_range[0], x_range[1]])
        y_vals1 = self.slopes[0] * x_vals + self.line1_c
        y_vals2 = self.slopes[1] * x_vals + self.line2_c
        return cp.asnumpy(x_vals), cp.asnumpy(y_vals1), cp.asnumpy(y_vals2)

    def calc_range(self, center_point: tuple, padding: int = 10) -> (tuple, tuple):
        # calculate x and y range with given points
        x_range = [min(self.points[0][0], self.points[1][0], center_point[0]) - padding,
                   max(self.points[0][0], self.points[1][0], center_point[0]) + padding]
        y_range = [min(self.points[0][1], self.points[1][1], center_point[1]) - padding,
                   max(self.points[0][1], self.points[1][1], center_point[1]) + padding]
        return x_range, y_range

    def draw(self, center_point: tuple, padding: int = 10):
        x_range, y_range = self.calc_range(center_point, padding)
        # Generate x values and corresponding y values for both lines
        x_vals, y_vals1, y_vals2 = self.calculate_line_values(x_range)

        # Plotting the points and lines
        fig, ax = plt.subplots()

        ax.plot(center_point[0], center_point[1], 'ro', label='Midpoint')  # Midpoint
        ax.plot(self.points[0][0], self.points[0][1], 'gx', label='line1_point')  # line1_point
        ax.plot(self.points[1][0], self.points[1][1], 'bx', label='line2_point')  # line2_point
        ax.plot(x_vals, y_vals1, 'g-', label='Line 1')  # Line 1
        ax.plot(x_vals, y_vals2, 'b-', label='Line 2')  # Line 2

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        ax.legend()
        ax.invert_yaxis()
        plt.show()

    def draw_on_image(self, center_point: tuple, image: np.ndarray, padding: int = 10):
        x_range, y_range = self.calc_range(center_point, padding)
        # Adjust ranges to fit within the image bounds
        min_x = max(int(x_range[0]), 0)
        max_x = min(int(x_range[1]), image.shape[1])
        min_y = max(int(y_range[0]), 0)
        max_y = min(int(y_range[1]), image.shape[0])
        cropped_image = image[min_y:max_y, min_x:max_x]
        img_height = cropped_image.shape[0]

        x_vals, y_vals1, y_vals2 = self.calculate_line_values([min_x, max_x])
        y_vals1 = [flip_y(y, min_y, img_height) for y in y_vals1]
        y_vals2 = [flip_y(y, min_y, img_height) for y in y_vals2]
        flipped_center_y = flip_y(center_point[1], min_y, img_height)

        fig, ax = plt.subplots()
        ax.imshow(cropped_image, extent=[min_x, max_x, min_y, max_y])

        ax.plot(x_vals, y_vals1, 'g-', label='Line 1', linewidth=2)
        ax.plot(x_vals, y_vals2, 'b-', label='Line 2', linewidth=2)
        ax.plot(center_point[0], flipped_center_y, 'ro', label='Midpoint')
        ax.plot(self.points[0][0], flip_y(self.points[0][1], min_y, img_height), 'gx', label='line1_point')
        ax.plot(self.points[1][0], flip_y(self.points[1][1], min_y, img_height), 'bx', label='line2_point')

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Line Distance Visualization on Image')
        ax.legend()
        ax.grid(True)
        plt.show()


class CircularDist(PointDistance):
    def __init__(self, center: tuple, x_len: float, y_len: float, slope: float):
        """

        :param center: (x, y) center of ellipse
        :param x_len: x axe length
        :param y_len: y axe length
        :param slope: slope of the ellipse
        """
        self.center = center
        self.x_len = x_len
        self.y_len = y_len
        if slope == 0:
            slope = 0.0001
        self.angle = math.atan(-1 / slope)
        self.ellipse = self.create_ellipse_cuda()

    def create_ellipse_cuda(self) -> Polygon:
        angle_rad = xp.radians(self.angle)
        t_values = xp.linspace(0, 360, num=180)  # Degree values for the ellipse points
        rad_values = xp.radians(t_values)  # Convert degrees to radians

        x = self.x_len * xp.cos(rad_values)
        y = self.y_len * xp.sin(rad_values)

        # Rotate points
        x_rotated = x * xp.cos(angle_rad) + y * xp.sin(angle_rad)
        y_rotated = x * xp.sin(angle_rad) + y * xp.cos(angle_rad)

        # Translate points to the center
        x_final = x_rotated + self.center[0]
        y_final = y_rotated + self.center[1]

        # Convert CuPy arrays back to NumPy arrays for Shapely, if using CuPy
        if USE_CUDA:
            ellipse_points = list(zip(cp.asnumpy(x_final), cp.asnumpy(y_final)))
        else:
            ellipse_points = list(zip(x_final, y_final))
        return Polygon(ellipse_points)

    def calc_dist(self, center_point: tuple) -> float:
        x_diff = self.center[0] - center_point[0]
        y_diff = self.center[1] - center_point[1]

        sin_theta = xp.sin(self.angle)
        cos_theta = xp.cos(self.angle)
        rotated_x_diff = x_diff * cos_theta + y_diff * sin_theta
        rotated_y_diff = y_diff * cos_theta - x_diff * sin_theta

        perpendicular_diff = rotated_x_diff / self.x_len
        parallel_diff = rotated_y_diff / self.y_len

        dist = xp.sqrt(perpendicular_diff ** 2 + parallel_diff ** 2)
        return xp.asnumpy(dist).item()
        # if rotated_y_diff > 0:
        #     return -dist

    def draw(self, center_point: tuple, padding: int = 10):
        # Plotting the points and lines
        fig, ax = plt.subplots()

        x, y = self.ellipse.exterior.xy
        ax.fill(x, y, alpha=0.5, color='blue', label='Ellipse')
        ax.plot(center_point[0], center_point[1], 'ro', label='Midpoint')
        ax.plot(self.center[0], self.center[1], 'go', label='Centerpoint')

        x.append(center_point[0])
        y.append(center_point[1])
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        ax.legend()
        ax.invert_yaxis()
        plt.show()

    def draw_on_image(self, center_point: tuple, image: np.ndarray, padding: int = 10):
        # Plotting the points and lines
        fig, ax = plt.subplots()

        x, y = self.ellipse.exterior.xy
        all_x, all_y = x[:], y[:]

        all_x.append(center_point[0])
        all_y.append(center_point[0])
        min_x = max(int(min(all_x) - padding), 0)
        max_x = min(int(max(all_x) + padding), image.shape[1])
        min_y = max(int(min(all_y) - padding), 0)
        max_y = min(int(max(all_y) + padding), image.shape[0])

        cropped_image = image[min_y:max_y, min_x:max_x]
        img_height = cropped_image.shape[0]

        ax.imshow(cropped_image, extent=[min_x, max_x, min_y, max_y])
        ax.fill(x, [flip_y(yi, min_y, img_height) for yi in y], alpha=0.5, color='blue', label='Ellipse')
        ax.plot(center_point[0], flip_y(center_point[1], min_y, img_height), 'ro', label='Midpoint')
        ax.plot(self.center[0], flip_y(self.center[1], min_y, img_height), 'go', label='Centerpoint')

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.legend()
        plt.show()
        plt.show()
