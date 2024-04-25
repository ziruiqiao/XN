import cv2
import numpy as np
from ActivePyTools.constants import *


def get_image_and_grey(img_path):
    img = cv2.imread(img_path)

    height = img.shape[0]
    resize_rate = DEFAULT_RESIZE_RATE
    while height * resize_rate < (1 / LINE_HEIGHT_PERCENT):
        resize_rate *= 2

    resized_image = cv2.resize(img, None, fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    return img, thresh, resize_rate


def check_color_area(grey_img, min_rows, min_cols, color=255):
    # Get the dimensions of the array
    rows, cols = grey_img.shape
    color_lines = []

    i = 0
    while i <= rows - min_rows:
        found_color_block = False  # Flag to check if we found a color block
        for j in range(cols - min_cols + 1):
            # Check if the subarray is filled with the target value
            if np.all(grey_img[i:i + min_rows, j:j + min_cols] == color):
                color_lines.append((i, j))  # Append the starting coordinates
                found_color_block = True
                break  # Break the inner loop
        if found_color_block:
            i += min_rows  # Skip min_rows rows
        else:
            i += 1  # Only increment i by 1 if no block was found

    if len(color_lines):
        return True, color_lines
    return False, []


def calculate_original_area(resized_top_left_x, resized_top_left_y, resized_bottom_right_x, resized_bottom_right_y,
                            resize_factor):
    # Calculate the top-left corner of the corresponding area in the original image
    original_top_left_x = resized_top_left_x * resize_factor
    original_top_left_y = resized_top_left_y * resize_factor
    original_bt_right_x = resized_bottom_right_x * resize_factor
    original_bt_right_y = resized_bottom_right_y * resize_factor

    # Calculate the bottom-right corner of the corresponding area in the original image
    original_bottom_right_x = original_bt_right_x + (resize_factor - 1)
    original_bottom_right_y = original_bt_right_y + (resize_factor - 1)

    return original_top_left_x, original_top_left_y, original_bottom_right_x, original_bottom_right_y


def get_remain_parts(im, blank_area, min_h):
    remain_part = []
    start_y = None
    end_y = None
    start = 0

    for area in blank_area:
        if start_y is None:
            start_y = area[0]
            end_y = area[2]
        else:
            if area[0] < end_y:
                end_y = area[2]
            else:
                diff = end_y - start_y
                remain_rate = (1 - CROP_PORTION_RATE) / 2
                remain_height = diff * remain_rate
                update_start_y = round(start_y + remain_height)
                update_end_y = round(end_y - remain_height)

                if start_y - 0 <= min_h:
                    start = update_end_y
                    pass
                else:
                    remain_part.append((start, update_start_y))
                    start = update_end_y

                start_y = None
                end_y = None
                continue

    if im.shape[0] - start > min_h:
        remain_part.append((start, im.shape[0]))
    return remain_part


def cut_image_horizontally(image, cutline):
    img_height, img_width, _ = image.shape

    parts = []

    for line in cutline:
        left = 0
        top = line[0]
        right = img_width
        bottom = line[1]
        part = image[top:bottom, left:right]
        parts.append(part)

    return parts


def cut_image_vertically(image, num_parts: int = VERTICAL_CROP_NUM, overlap_percent: int = OVERLAP_PERCENT):

    if num_parts == 1:
        return [image]

    img_height, img_width, _ = image.shape
    overlap = img_width * overlap_percent / 100
    total_divisible_width = img_width + (overlap * (num_parts - 1))
    part_width = total_divisible_width // num_parts
    parts = []

    for i in range(num_parts):
        left = max(0, i * part_width - i * overlap)
        top = 0
        right = ((i + 1) * part_width - i * overlap) if i < num_parts - 1 else img_width
        # End of each part (normal end unless it's the last part, then ensure it ends at the image's bottom)
        bottom = img_height

        part = image[top:bottom, int(left):int(right)]
        parts.append(part)

    return parts


def check_shelf_divider(location, shape, divider_rate: float = BOOKSHELF_DIVIDER_RATE):
    height, width = shape
    start_y = width * (1 - divider_rate) / 2
    end_y = height - start_y

    for loc in location:
        if start_y < loc[0] < end_y:
            return True
    return False


def crop_image(img_path):
    im, thresh, resize_rate = get_image_and_grey(img_path)
    image_height, image_width = thresh.shape
    min_line_height = image_height * LINE_HEIGHT_PERCENT
    min_line_width = image_width * LINE_WIDTH_PERCENT
    min_rows = round(min_line_height)
    min_cols = round(min_line_width)
    cropped_img = []

    white_exists, white_location = check_color_area(thresh, min_rows, min_cols)
    black_exists, black_location = check_color_area(thresh, min_rows, min_cols, 0)

    location = white_location if len(white_location) > len(black_location) else black_location
    exists = white_exists if len(white_location) > len(black_location) else black_exists

    if not exists or not check_shelf_divider(location, thresh.shape):
        print('========================================================')
        print('No need to be cropped')
        print('========================================================')
        cropped_img.append([im])

    else:
        min_height = resize_rate/DEFAULT_RESIZE_RATE/LINE_HEIGHT_PERCENT * 2
        color_area = []

        for loc in location:
            original_area = calculate_original_area(loc[0], loc[1], loc[0] + min_rows, loc[1] + min_cols, 1/resize_rate)
            color_area.append(original_area)

        remain_parts = get_remain_parts(im, color_area, min_height)
        horizontal_parts = cut_image_horizontally(im, remain_parts)

        for part in horizontal_parts:
            vertical_parts = cut_image_vertically(part)
            cropped_img.append(vertical_parts)

    return im, cropped_img
