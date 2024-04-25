from typing import Tuple, Any
from PIL import Image
import io
from google.cloud import vision, vision_v1
from botocore import client
import pandas as pd
from scipy.spatial import distance

from ActivePyTools.utils import ensure_string


def get_google_response(im, cli: vision_v1.ImageAnnotatorClient) -> tuple[Any, Any]:
    """
    Get Google Vision Response on the given image.
    :param im: Given image
    :param cli: Google Api Initialized
    :return:
       texts: separated Text and locations
       full_text: Google grouped sentences
    """
    im_pil = Image.fromarray(im)
    buffer = io.BytesIO()
    im_pil.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    img = vision.Image(content=image_bytes)

    text_response = cli.text_detection(image=img)
    texts = text_response.text_annotations
    response = cli.document_text_detection(image=img)
    full_text = response.full_text_annotation

    return texts, full_text


def get_amazon_response(im, rekog: client.Rekognition) -> tuple[Any, Any]:
    """
    Get Amazon Rekognition Response on the given image.
    :param im: Given image
    :param rekog: Amazon Api Initialized
    :return:
       words: separated Text and locations
       lines: Amazon grouped sentences
    """
    im_pil = Image.fromarray(im)
    buffer = io.BytesIO()
    im_pil.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    resp = rekog.detect_text(
        Image={'Bytes': image_bytes}
    )
    lines = [e for e in resp['TextDetections'] if e['Type'] == 'LINE']
    words = [e for e in resp['TextDetections'] if e['Type'] == 'WORD']

    return words, lines


def google_extract_text_and_position(ocr_result, img_shape: tuple) -> (str, pd.DataFrame):
    """
    format google vision response
    :param ocr_result: original google vision response
    :param img_shape: given image's shape (height, width, RGB)
    :return:
        allText: Google grouped sentences
        df: Pandas Dataframe in with columns[txt, confidence, vertices, boundBox]
    """
    if not ocr_result:
        return None, pd.DataFrame()

    allText = ocr_result[0].description if ocr_result else ""
    height, width, _ = img_shape

    text_list = []
    for idx, item in enumerate(ocr_result[1:]):
        vertices = [(v.x, v.y) for v in item.bounding_poly.vertices]

        min_x = min(vertices, key=lambda point: point[0])[0]
        max_x = max(vertices, key=lambda point: point[0])[0]
        min_y = min(vertices, key=lambda point: point[1])[1]
        max_y = max(vertices, key=lambda point: point[1])[1]

        item_dict = {
            'txt': item.description,
            'confidence': item.confidence if item.confidence != 0.0 else 100,
            'vertices': vertices,
            'boundBox': {
                "Width": max_x - min_x,
                "Height": max_y - min_y,
                "Left": min_x,
                "Top": min_y
            }
        }

        text_list.append(item_dict)
    df = pd.DataFrame(text_list)

    return allText, df


def amazon_extract_text_and_position(ocr_result, img_shape: tuple) -> (str, pd.DataFrame):
    """
    format google vision response
    :param ocr_result: original Amazon Rekognition Response
    :param img_shape: given image's shape (height, width, RGB)
    :return:
        allText[:-1]: Amazon grouped sentences
        df: Pandas Dataframe in with columns[txt, confidence, vertices, boundBox]
    """
    text_list = []
    allText = ''

    height, width, _ = img_shape

    for idx, item in enumerate(ocr_result):
        vertices = [(round(v['X'] * width, 3), round(v['Y'] * height, 3)) for v in item['Geometry']['Polygon']]
        box = item['Geometry']['BoundingBox']
        for key in box:
            if key == 'Width' or key == 'Left':
                box[key] = round(box[key] * width, 3)
            elif key == 'Height' or key == 'Top':
                box[key] = round(box[key] * height, 3)

        item_dict = {
            'txt': item['DetectedText'],
            'confidence': round(item['Confidence']),
            'vertices': vertices,
            'boundBox': box
        }
        text_list.append(item_dict)
        allText += item_dict['txt'] + '\n'
    df = pd.DataFrame(text_list)

    return allText[:-1], df


def analyze_text_location(text_collection: pd.DataFrame, img_shape: tuple) -> pd.DataFrame:
    """
    Provide Simple Analysis to text locations
    :param text_collection: dataframe with columns[txt, confidence, vertices, boundBox]
        Note: vertices column contains 4 points and the order is always
          1. left top of word when it can be read
          2. right top of word when it can be read
          3. right bottom of word when it can be read
          4. left bottom of word when it can be read
    :param img_shape: given image's shape (height, width, RGB)
    :return:
        texts: dataframe with new columns [slopes, font, word_len, direction, center_point]
    """
    texts = text_collection.copy()
    height, width, _ = img_shape
    img_ratio = height / width

    for column in ['slopes', 'font', 'word_len', 'direction', 'center_point']:
        if column not in texts.columns:
            texts[column] = None

    for idx, text in texts.iterrows():
        vertices = text['vertices']

        slopes = []

        for i in range(len(vertices)):
            width = vertices[i][0] - vertices[(i + 1) % len(vertices)][0]
            height = vertices[(i + 1) % len(vertices)][1] - vertices[i][1]

            # Calculate the slope, handling the case where width is zero
            slope = float(1000) if width == 0 else height / width
            slopes.append(round(slope, 3))

        texts.at[idx, 'slopes'] = tuple(slopes)
        texts.at[idx, 'font'] = (
                                        distance.euclidean(vertices[1], vertices[2]) +
                                        distance.euclidean(vertices[3], vertices[0])
                                ) / 2
        texts.at[idx, 'word_len'] = (
                                            distance.euclidean(vertices[0], vertices[1]) +
                                            distance.euclidean(vertices[2], vertices[3])
                                    ) / 2
        texts.at[idx, 'direction'] = 'vertical' if abs(slopes[0]) > 1 else 'horizontal'
        x_cords, y_cords = [point[0] for point in vertices], [point[1] for point in vertices]

        texts.at[idx, 'center_point'] = (sum(x_cords) / len(x_cords), sum(y_cords) / len(y_cords))
    return texts


def collect_human_eye_book_names(shelf: pd.DataFrame) -> tuple[list, list]:
    """
    Reformat dataframe to lists for performance measurement.
    :param shelf: dataframe of human saw book names and authors
        with columns[Row, No., Book Name, Author Name]
    :return:
        splits: list of all words of Book Names and Author Names.
        names: well grouped Book Names and Author Names
    """
    names = []
    for index, row in shelf.iterrows():
        book_name = row['Book Name']
        author_name = row['Author Name']
        temp_name = ''
        if not pd.isna(book_name):
            temp_name += book_name
        if not pd.isna(author_name):
            temp_name += ' ' + author_name
        if temp_name != '':
            names.append(temp_name)
    splits = []
    for string in names:
        splits.extend(ensure_string(string).split())
    return splits, names


def eval_object_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate string expressions stored in object-type columns of a DataFrame,
    excluding specific columns like 'txt' and 'direction'. Each string expression
    is evaluated using a safe method that handles errors.

    :param dataframe: Pandas DataFrame with columns containing string expressions.
    :return: DataFrame with the object-type columns evaluated, where possible.

    Example usage:
    >>> df = pd.DataFrame({'A': ['1+1', '2+2'], 'txt': ['ignore', 'ignore']})
    >>> eval_object_columns(df)
       A      txt
    0  2  ignore
    1  4  ignore
    """

    def try_eval(item):
        try:
            return eval(item)
        except Exception:
            return item

    object_columns = [col for col in dataframe.select_dtypes(include=['object']).columns if
                      col not in ['txt', 'direction']]
    for column in object_columns:
        dataframe[column] = dataframe[column].apply(lambda item: try_eval(item))

    return dataframe


def load_previous_df(path: str) -> (None, pd.DataFrame):
    """
    use previously stored dataframe
    usually with columns:
    [txt, confidence, vertices, boundBox, slopes, font, word_len, direction, center_point]
    :param path: csv file path
    :return: previously stored dataframe
    """
    temp_df = pd.read_csv(path)
    evaled_df = eval_object_columns(temp_df)
    # related_text = group_related_text(evaled_df)
    # related_text = sort_and_trim_dict(related_text)
    # filtered_data = remove_duplicates(related_text)
    return None, evaled_df


def collect_book_names(img, api, select_ocr: str) -> tuple[str, pd.DataFrame]:
    if select_ocr == 'google':
        words, lines = get_google_response(img, api)
        allText, text_df = google_extract_text_and_position(words, img.shape)
    elif select_ocr == 'amazon':
        words, lines = get_amazon_response(img, api)
        _, text_df = amazon_extract_text_and_position(words, img.shape)
        allText, _ = amazon_extract_text_and_position(lines, img.shape)
    else:
        print("=========================")
        print("Invalid API")
        print("=========================")
        return None, None, None

    text_df = analyze_text_location(text_df, img.shape)
    # related_text = group_related_text(text_df)
    # filtered_data = remove_duplicates(related_text)
    return  allText, text_df
