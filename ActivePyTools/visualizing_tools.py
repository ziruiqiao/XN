import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.core.series import Series


def plot_box_intersect(vertices, slopes, intercepts, additional_vertices=None, padding=10, y_limits=None):
    # Plotting
    if additional_vertices is None:
        additional_vertices = []
    x_poly, y_poly = zip(*vertices + [vertices[0]])  # Close the polygon
    x_range_min = min(x_poly + tuple(v[0] for v in additional_vertices)) - padding
    x_range_max = max(x_poly + tuple(v[0] for v in additional_vertices)) + padding
    x_line = range(x_range_min, x_range_max)
    y_line1 = [slopes[0] * x + intercepts[0] for x in x_line]
    y_line2 = [slopes[1] * x + intercepts[1] for x in x_line]

    plt.figure(figsize=(8, 6))
    plt.plot(x_poly, y_poly, label='Polygon', color='blue')  # Plot polygon
    plt.plot(x_line, y_line1, label=f'Line 1: y = {slopes[0]}x + {intercepts[0]}', color='red')  # Plot first line
    plt.plot(x_line, y_line2, label=f'Line 2: y = {slopes[1]}x + {intercepts[1]}', color='green')  # Plot second line
    if additional_vertices:
        plt.scatter(*zip(*additional_vertices), color='orange', s=100,
                    label='Additional Vertices')  # Plot additional vertices as large bubbles
    plt.title('Intersection of Polygon and Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if y_limits:
        plt.ylim(y_limits)
    plt.show()


def get_rect(coord: dict) -> dict:
    return {
        'x': coord['Left'],
        'y': coord['Top'],
        'width': coord['Width'],
        'height': coord['Height']
    }


def get_text_data(text: Series) -> dict:
    coord = text.boundBox
    return get_rect(coord)


def draw_rectangle(rectangle: dict, color='white') -> patches.Rectangle:
    rect = patches.Rectangle((rectangle['x'],
                              rectangle['y']),
                             rectangle['width'],
                             rectangle['height'],
                             linewidth=2,
                             edgecolor=color, facecolor='none')
    return rect


def plot_text_detected(im, df: pd.DataFrame, save_picname: str, dpi=300):
    if df is not None:
        all_text_rects = [draw_rectangle(get_text_data(annotation), color='red')
                          for annotation in df.itertuples()]
    else:
        all_text_rects = []

    fix, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(im)

    for rect in all_text_rects:
        ax.add_patch(rect)

    if save_picname is not None:
        plt.savefig(save_picname, dpi=dpi)
    else:
        plt.show()
    plt.close(fix)
