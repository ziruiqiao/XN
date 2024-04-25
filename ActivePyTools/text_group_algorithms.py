import pandas as pd
from pandas.core.series import Series
from sklearn.cluster import KMeans

from ActivePyTools.constants import CONFIDENCE_THRESHOLD
from ActivePyTools.utils import *
from ActivePyTools.text_geometry_analysis_algorithm import *
from ActivePyTools.metrics import calculate_text_match_percentage
import ActivePyTools.constants as c


def sort_and_trim_dict(dict_data: dict, n: int = 5) -> dict:
    """
    For each list in the dictionary, sort the tuples by the last element
    and trim the list to keep only the first n tuples.
    :param dict_data: Dict -- The input dictionary with list of tuples as values.
    :param n: Int -- Number of tuples to keep in each list after sorting.
    :return dict: The updated dictionary with sorted and trimmed lists.
    """

    def custom_sort_key(tup):
        # If tuple has more than 3 elements, use the last element for sorting
        if len(tup) > 3:
            return tup[-1]
        else:
            # Otherwise, use the last element of the 3rd item (assuming it's a tuple)
            return tup[2][-1]

    for key in dict_data:
        # Sort the list of tuples using the custom sort function and trim it
        sorted_list = sorted(dict_data[key], key=custom_sort_key)[:n]
        dict_data[key] = sorted_list

    return dict_data


def criteria_selector(base_point: Series, other_point: Series,
                      iv: ImportVariables) -> float:
    """
    Determines and calculates the relationship between two points based on the specified criteria.
    This function supports both distance and area calculations, adapting to either circular or linear
    geometric relationships defined by enums.
    :param base_point: A row of df -- containing details like center point, slopes, vertices and direction.
    :param other_point: A row of df -- for the comparison; contains center point and vertices.
    :param iv: ImportVariables -- Necessary variables includes xdt, ydt, img_shape and metric.
    :return float: The calculated distance or area ratio, depending on the selected metric.

    :raise ValueError: If an invalid metric type is provided.

    This function first determines the type of calculation based on the metric, configures the geometric
    dimensions and slopes, then calculates either the distance or the area based on the metric's criteria.
    It handles both circular and linear relationships using a unified approach.
    """
    xdt, ydt, img_shape, metric = iv.get_variables()
    if not isinstance(metric, (DistanceCriteria, AreaCriteria)):
        raise ValueError("Invalid Metric Type Provided")

    if metric.type == 'Circular':
        x_len, y_len = calc_xy_len(base_point, xdt, xdt if metric.equal_dims else ydt)
        slope = select_slope(base_point.slopes, base_point.direction)

        shape_obj = CircularDist(base_point.center_point, x_len, y_len, slope) \
            if 'Distance' in metric.name else \
            CircularIntersection(base_point.center_point, x_len, y_len, slope)

    elif metric.type == 'Linear':
        top_left, _, _, top_right = find_extreme_vertices(base_point.vertices)
        direction_slopes = base_point.slopes[::2] if base_point.direction == "vertical" else base_point.slopes[1::2]

        shape_obj = LinearDist((top_left, top_right), direction_slopes) \
            if 'Distance' in metric.name else \
            LinearIntersection.from_slopes_and_intercepts((top_left, top_right), direction_slopes, img_shape)
    else:
        raise Exception(metric.description)

    if 'Distance' in metric.name:
        return shape_obj.calc_dist(other_point.center_point)
    else:
        area = shape_obj.intersection_area(other_point.vertices)
        return area / (other_point.font ** 2)


def remove_duplicates(original_data: dict) -> dict:
    """
    remove 100% duplicated key and value from the dictionary
    :param original_data: dict -- a dictionary contains related text
    :return: deduplicated new dictionary
    """
    data = original_data.copy()
    duplicate_data_key = []

    for key1, key1_value_list in data.items():
        key1_words = [item[1] for item in key1_value_list]  # key1 related words
        for key2, key2_value_list in data.items():
            if key2 <= key1:  # prevent duplicate analysis
                continue
            key2_words = [item[1] for item in key2_value_list]  # key2 related words

            set2 = set(key2_words)  # put word list to Set for easier analysis
            set1 = set(key1_words)
            if set2.issubset(set1):  # record key2 to remove if it is subset of key1
                duplicate_data_key.append(key2)
            elif set1.issubset(set2):  # record key1 to remove if it is subset of key2
                duplicate_data_key.append(key1)
                break  # analyze next key

    for k in duplicate_data_key:
        data.pop(k, None)  # remove recorded keys

    return data


def group_related_text(texts: pd.DataFrame, iv: ImportVariables) -> dict:
    """
    group related text together base on given metric
    :param texts: Dataframe -- contains all text
    :param iv: ImportVariables -- Necessary variables includes xdt, ydt, img_shape and metric.
    :return: dict -- related text
        key: index of base text
        value: list of related text tuple, each contains (index, text, distance)
               except for the first text (key itself)
    """
    related_text = {}
    for txt1_idx, txt1 in texts.iterrows():
        if txt1['confidence'] <= CONFIDENCE_THRESHOLD:
            continue
        related_text[txt1_idx] = []
        related_text[txt1_idx].append([txt1_idx, txt1['txt']])
        for txt2_idx, txt2 in texts.iterrows():
            if txt2['confidence'] <= CONFIDENCE_THRESHOLD or txt1_idx == txt2_idx:
                continue
            # print(f"t1: {t1_idx} {t1.txt}; t2: {t2_idx} {t2.txt}")
            distance = criteria_selector(txt1, txt2, iv)
            # inter_points = check_points(t2_polygon, slope1, p1, p2, img)

            if (distance > 0 and isinstance(iv.metric, AreaCriteria)) or (
                    distance < 1 and isinstance(iv.metric, DistanceCriteria)):
                content = (txt2_idx, txt2['txt'], distance)
                # Ensure no duplicate entries are added
                if content not in related_text[txt1_idx]:
                    related_text[txt1_idx].append(content)
    return related_text


def update_distance(element2: tuple, element1: tuple, df_all: pd.DataFrame, iv: ImportVariables) -> float:
    """

    :param element2: tuple -- (index, text, distance)
    :param element1: tuple -- (index, text, distance)
    :param df_all: dataframe -- contains all text info
    :param iv: ImportVariables -- Necessary variables includes xdt, ydt, img_shape and metric.
    :return: updated criteria value
    """
    t1 = df_all.iloc[element1[0]]
    t2 = df_all.iloc[element2[0]]
    return criteria_selector(t1, t2, iv)


def combine_elements(data_dict, df_all, iv: ImportVariables) -> dict:
    """
    combine text groups if recall and precision of the two are larger than threshold.
    Remove duplicated text groups after combining.
    :param data_dict: dict -- text groups
    :param df_all: Dataframe -- df of all texts
    :param iv: ImportVariables -- Necessary variables includes xdt, ydt, img_shape and metric.
    :return: data: dict -- data_dict after combination and deduplication
    """
    keys_to_remove = set()
    data = data_dict.copy()
    for key1 in data:
        key1_itself = data[key1][0]
        for key2 in data:
            if key1 < key2 and key1 not in keys_to_remove:
                precision, recall, shared_elements = calculate_text_match_percentage(data[key1], data[key2])

                if recall >= c.recall_threshold and precision >= c.precision_threshold:
                    key2_only_items = [item for item in data[key2] if item[0] not in [i[0] for i in data[key1]]]

                    # Update elements that are newly added from the smaller key's list
                    for index, key2_only_item in enumerate(key2_only_items):
                        if key2_only_item[0] not in shared_elements and key2_only_item[0] != key1_itself[0]:
                            # Update the 4th element using the update_distance function
                            # Assume point1, width, height, and slope are defined elsewhere
                            temp_lst = list(key2_only_item)
                            updated_dist = update_distance(key2_only_item, key1_itself, df_all, iv)
                            if len(temp_lst) == 3:
                                temp_lst[2] = updated_dist
                            else:
                                temp_lst.append(updated_dist)
                            key2_only_items[index] = tuple(temp_lst)

                    combined_list = data[key1] + key2_only_items

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


# ========================
# Unrevised Function
# ========================

def test_clusters(txt_df, n_cluster, xdt, ydt):
    divide_by_xydt = lambda point: (point[0] / xdt, point[1] / ydt)
    pos_lst = []
    for index, row in txt_df.iterrows():
        transformed_tuples = [divide_by_xydt(tup) for tup in row['vertices']]
        flattened = np.concatenate(transformed_tuples)  # Flatten and combine tuples
        pos_lst.append(flattened)

    pos_arr = np.array(pos_lst)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(pos_arr)
    labels = kmeans.labels_

    # plt.figure(figsize=(8, 6))
    # for i in range(n_cluster):
    #     cluster = pos_arr[labels == i]
    #     plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='*')
    # plt.title('Text Clustering by Position')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    cluster_labels = kmeans.predict(pos_arr)
    df = txt_df.copy()
    df['Cluster'] = cluster_labels

    book_names = []
    for i in range(n_cluster):
        selected_cluster = i
        filtered_df = df[df['Cluster'] == selected_cluster]
        txt_list = filtered_df['txt'].tolist()
        book_names.append(txt_list)
    return book_names
