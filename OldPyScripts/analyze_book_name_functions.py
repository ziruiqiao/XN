
import matplotlib.pyplot as plt
import math
from math import sqrt
import requests
import pandas as pd
from ActivePyTools.constants import *
from ActivePyTools.grab_data import *


lines_to_plot = []


def plot_line(img, m, b):
    height, width, _ = img.shape
    x_start, x_end = 0, width
    y_start = m * x_start + b
    y_end = m * x_end + b

    # Adjust y_start and y_end if they are outside the image boundaries
    if y_start < 0:
        y_start = 0
        x_start = (y_start - b) / m
    elif y_start > height:
        y_start = height
        x_start = (y_start - b) / m

    if y_end < 0:
        y_end = 0
        x_end = (y_end - b) / m
    elif y_end > height:
        y_end = height
        x_end = (y_end - b) / m

    result = [x_start, x_end], [y_start, y_end]
    lines_to_plot.append(result)


def is_point_within_middle(x, y, s, b1, b2, img):
    y1 = s * x + b1
    y2 = s * x + b2
    if DEBUG:
        plot_line(img, s, b1)
        plot_line(img, s, b2)
    return min(y1, y2) <= y <= max(y1, y2)


def check_points(points, s1, P1, P2, img):
    points = [{'X': point[0], 'Y': point[1]} for point in points]

    b1 = P1[1] - s1 * P1[0]
    b2 = P2[1] - s1 * P2[0]

    counter = 0
    for point in points:
        if is_point_within_middle(point['X'], point['Y'], s1, b1, b2, img):
            counter += 1

    left_side = 0
    right_side = 0
    # Check if the counter is 0 before proceeding
    if counter != 4:
        for point in points:
            # Calculate the y-values using the given slope (s1) and intercepts (b1, b2) for each X
            y_value_1 = s1 * point['X'] + b1
            y_value_2 = s1 * point['X'] + b2

            # Determine the position of the point relative to the lines
            if point['Y'] <= min(y_value_1, y_value_2):
                left_side += 1
            if point['Y'] >= max(y_value_1, y_value_2):
                right_side += 1
    if left_side == 2 and right_side == 2:
        return 4
    elif left_side == 1 and right_side == 2 or left_side == 2 and right_side == 1:
        return 3
    return counter


def distance_relative_wh(point1, point2, width, height, slope):
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    if slope == 0:
        slope = 0.0001
    theta = math.atan(-1/slope)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    rotated_x_diff = x_diff * cos_theta + y_diff * sin_theta
    rotated_y_diff = y_diff * cos_theta - x_diff * sin_theta

    perpen_diff = rotated_x_diff / (width * X_DISTANCE_THRESHOLD)
    parall_diff = rotated_y_diff / (height * Y_DISTANCE_THRESHOLD)

    dist = sqrt(perpen_diff ** 2 + parall_diff ** 2)
    # if rotated_y_diff > 0:
    #     return -dist
    return dist


def group_related_text(texts):
    related_text = {}
    for t1_idx, t1 in texts.iterrows():
        if t1['confidence'] <= CONFIDENCE_THRESHOLD:
            continue
        related_text[t1_idx] = []
        related_text[t1_idx].append((t1_idx, t1['txt'], t1.mid_point, 0.0))
        for t2_idx, t2 in texts.iterrows():
            if t2['confidence'] <= CONFIDENCE_THRESHOLD or t1_idx == t2_idx:
                continue
            t1_slope = t1['slopes']
            t1_direction = t1['direction']
            t1_polygon = t1['vertices']
            t2_polygon = t2['vertices']
            v_slope = -200 if t1_slope[0] == 0 else -1/t1_slope[0]
            slope = min(abs(v_slope), abs(t1_slope[1]))
            if "vertical" == t1_direction:
                slope1 = -1 / slope if slope != 0 else -200
                p2 = t1_polygon[0]
                p1 = t1_polygon[3]
            elif "horizontal" == t1_direction:
                slope1 = slope
                p1 = t1_polygon[0]
                p2 = t1_polygon[1]
            else:
                print("===========================")
                print("l1 Invalid Direction")
                print("===========================")
                break
            distance = distance_relative_wh(t1.mid_point, t2.mid_point, t1.width, t1.height, slope1)
            # inter_points = check_points(t2_polygon, slope1, p1, p2, img)
            if distance < 1: # or inter_points >= 2:
                content = (t2_idx, t2['txt'], t2.mid_point, distance)
                if content not in related_text[t1_idx]:
                    related_text[t1_idx].append(content)
    return related_text


def remove_duplicates(original_data):
    data = original_data.copy()
    duplicate_data_key = []
    k1_duplicate = False

    if DEBUG:
        print(data)

    for k1, value_list in data.items():
        k1_words = [item[1] for item in value_list]
        for k2, value_list2 in data.items():
            if k2 <= k1:
                continue
            k2_words = [item[1] for item in value_list2]

            set2 = set(k2_words)
            set1 = set(k1_words)
            if set2.issubset(set1):
                if DEBUG:
                    print(k1, ': ', k1_words)
                    print(k2, ': ', k2_words)
                duplicate_data_key.append(k2)
            elif set1.issubset(set2):
                if DEBUG:
                    print(k1, ': ', k1_words)
                    print(k2, ': ', k2_words)
                duplicate_data_key.append(k1)
                break

    for k in duplicate_data_key:
        data.pop(k, None)

    return data


def update_distance(element2, element1, df_all):
    # Assuming width and height are known or calculated previously
    # You may need to adjust the logic to calculate or retrieve width and height
    point1 = element1[2]
    point2 = element2[2]
    t1 = df_all.iloc[element1[0]]
    t1_slope = t1['slopes']
    t1_direction = t1['direction']
    if "vertical" == t1_direction:
        slope = t1_slope[0]
    elif "horizontal" == t1_direction:
        slope = t1_slope[1]
    else:
        slope = 0
    return distance_relative_wh(point1, point2, t1.width, t1.height, slope)


def combine_elements(data_dict, df_all):
    keys_to_remove = set()
    data = data_dict.copy()
    for key1 in data:
        point1 = data[key1][0]
        for key2 in data:
            if key1 < key2 and key1 not in keys_to_remove:
                set1 = set([item[0] for item in data[key1]])
                set2 = set([item[0] for item in data[key2]])

                shared_elements = set1 & set2
                if shared_elements:
                    # Combine lists and update elements
                    # print(f"{key1}: {key2}")
                    combined_list = [item for item in data[key2] if item[0] not in [i[0] for i in data[key1]]]

                    # Update elements that are newly added from the smaller key's list
                    for index, item in enumerate(combined_list):
                        if item[0] not in shared_elements and item[0] != point1[0]:
                            # Update the 4th element using the update_distance function
                            # Assume point1, width, height, and slope are defined elsewhere
                            temp_lst = list(item)
                            updated_dist = update_distance(item, point1, df_all)
                            if len(temp_lst) == 4:
                                temp_lst[3] = updated_dist
                            else:
                                temp_lst.append(updated_dist)
                            combined_list[index] = tuple(temp_lst)

                    combined_list = data[key1] + combined_list

                    # Deduplicate the combined list based on the first element of each tuple
                    unique_combined_list = []
                    seen = set()
                    for item in combined_list:
                        if item[0] not in seen:
                            unique_combined_list.append(item)
                            seen.add(item[0])

                    data[key1] = unique_combined_list
                    keys_to_remove.add(key2)

    # Remove keys
    for key in keys_to_remove:
        del data[key]

    return data


def sort_book_name_candidates(text_dict: dict):
    book_name_lst = []
    data = text_dict.copy()
    return data, book_name_lst

    # for text1_idx, sublist in text_dict.items():
    #     if not sublist:
    #         continue
    #     sorted_data = sorted(sublist, key=lambda x: x[0][2][1])
    #     data[text1_idx] = sorted_data
    #
    #     if DEBUG:
    #         print(sorted_data)
    #
    #     # largest = sorted_data[-1]
    #     # name = ' '.join([v[1] for v in sorted_data[:-1]])
    #     # book_name_lst.append((name, largest[1]))
    #     name = ' '.join([v[1] for v in sorted_data])
    #     book_name_lst.append(name)
    #
    # return data, book_name_lst


def get_text_midpoint(text, df):
    for d in df.itertuples():
        if d.txt == text:
            return d.mid_point
    return 0, 0


def show_pic(img):
    fix, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # for p in polygons:
    # ax.add_patch(p)

    lst_no_duplicates = []
    [lst_no_duplicates.append(x) for x in lines_to_plot if x not in lst_no_duplicates]
    for l in lst_no_duplicates:
        ax.plot(l[0], l[1], color='red', linewidth=1)
    lines_to_plot.clear()

    plt.show()


def collect_book_names(img, api, select_ocr, use_line):
    if select_ocr == 'google':
        words, lines = get_google_response(img, api)
        allText, text_df = google_extract_text_and_position(words, img)
    elif select_ocr == 'amazon':
        words, lines = get_amazon_response(img, api)
        # text = lines if use_line else words
        _, text_df = amazon_extract_text_and_position(words, img)
        allText, _ = amazon_extract_text_and_position(lines, img)
    else:
        print("=========================")
        print("Invalid API")
        print("=========================")
        return None, None, None

    text_df = analyze_text_location(text_df, img)
    related_text = group_related_text(text_df)
    filtered_data = remove_duplicates(related_text)
    # data = combine_elements(filtered_data, text_df)
    # name_lst = [[e[1] for e in v] for k, v in data.items()]
    #
    # if DEBUG:
    #     show_pic(img)
    #     print(related_text)
    #     print(filtered_data)
    #     pass
    #
    # sorted_lst, book_name_lst = sort_book_name_candidates(filtered_data)
    #
    # if DEBUG:
    #     print(book_name_lst)
    return filtered_data, text_df, allText


def request_book_info(book_name_list):
    items = []

    for book_name in book_name_list:

        # author = author.replace(' ', '+')
        book_name = book_name.replace(' ', '+')
        # resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?"
        #                     f"q=intitle:{book_name}+inauthor:{author}").json()
        # book_lst_len = resp.get('totalItems')
        # search = book_name + '  ' + author
        # if book_lst_len == 0:
        #     resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?"
        #                         f"q=intitle:{author}+inauthor:{book_name}").json()
        #     book_lst_len = resp.get('totalItems')
        #     search = author + '  ' + book_name
        # if book_lst_len == 0:
        resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?"
                            f"q=intitle:{book_name}").json()
        book_lst_len = resp.get('totalItems')
        search = book_name
        # if book_lst_len == 0:
        #     resp = requests.get(f"https://www.googleapis.com/books/v1/volumes?"
        #                         f"q=intitle:{author}").json()
        #     book_lst_len = resp.get('totalItems')
        #     search = author
        print(book_lst_len, ' Searches: ', search)
        if book_lst_len != 0:
            item = [item['volumeInfo'] for item in resp.get('items')]
            item_df = pd.DataFrame(item)
            items.append(item_df)
    return items

# %%
