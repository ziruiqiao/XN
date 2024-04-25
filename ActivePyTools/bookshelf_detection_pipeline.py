import json, boto3

from ActivePyTools.metrics import *
from ActivePyTools.grab_data import *
from ActivePyTools.visualizing_tools import *
from ActivePyTools.crop_functions import *


def init_clients():
    """
    prepare amazon and google api client objects
    :return: amazon api, google api
    """
    with open('credentials.json') as f:
        credentials = json.load(f)

    aws_access_key_id = credentials['aws_access_key_id']
    aws_secret_access_key = credentials['aws_secret_access_key']
    rekognition = boto3.client(
        'rekognition',
        region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    client = vision.ImageAnnotatorClient()

    return rekognition, client


def get_cropped_images(img_path_lst: list, enable_print: bool = False) -> list:
    """
    crop list of image to smaller pieces
    :param img_path_lst: list -- paths of images
    :param enable_print: bool -- show picture or not
    :return: cropped_pics: list -- list of tuple (list_of_crop, original image)
            list_of_crop with shape (image_rows, image_columns, img),
            note: img is type of numpy.ndarray with shape (height, width, 3)
    """
    cropped_pics = []
    for idx, path in enumerate(img_path_lst):
        im, cropped_img = crop_image(path)
        cropped_pics.append((cropped_img, im))

        if enable_print:
            for idx2, img_row in enumerate(cropped_img):
                for idx3, image in enumerate(img_row):
                    fix,ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(image)
                    plt.show()
                    plt.close(fix)
    return cropped_pics


def get_text_location_from_api(
        img_path_lst: list,
        actual_data: list,
        cropped_pics: list,
        api, api_provider: str,
        enable_print: bool = False) -> dict:
    """
    Get text and location from api
    :param img_path_lst: list -- list of image paths
    :param actual_data: list -- list of list [list of word, list of sentences] (True Data)
    :param cropped_pics: list -- list of tuple (list_of_crop, original image)
            list_of_crop with shape (image_rows, image_columns, img),
            note: img is type of numpy.ndarray with shape (height, width, 3)
    :param api:
    :param api_provider: str -- api provider name for better print
    :param enable_print: bool -- print or not
    :return: text_df_dict: dict -- contains all text_df
    """

    if enable_print:
        print('=========================================================')
        print(f'{api_provider.upper()} OCR: \n')
    text_df_dict = {}
    for idx, (crops, ori_img) in enumerate(cropped_pics):
        path = img_path_lst[idx]
        img_name = path.split("/")[-1]
        text_df_dict[idx] = pd.DataFrame()

        detected_texts = []
        ocr_book_names = ''
        for idx2, img_row in enumerate(crops):
            for idx3, img_col in enumerate(img_row):
                allText, text_df = collect_book_names(img_col, api, api_provider)
                if allText is not None:
                    ocr_book_names += allText+'\n'
                if not text_df.empty:
                    # Save Cropped images
                    pic_name = f'{api_provider}/{img_name}_{idx2}_{idx3}.jpg'
                    cv2.imwrite(f'./docs/crop_pics/{pic_name}', img_col)
                    # Save Cropped images with all detected text labelled
                    plot_text_detected(img_col, text_df, f'./docs/detected_text_pics/{pic_name}')
                    # Save Detected text
                    detected_texts = detected_texts + text_df['txt'].tolist()
                    # Label the which cropped image does this text_df belong to
                    text_df['crop_idx'] = [(idx2, idx3)] * len(text_df)
                    text_df['Left'] = text_df['boundBox'].apply(lambda x: x['Left'])
                    text_df['Top'] = text_df['boundBox'].apply(lambda x: x['Top'])
                    df_sorted = text_df.sort_values(by=['Left', 'Top'])
                    # concat the text_df to the df of this picture
                    text_df_dict[idx] = pd.concat([text_df_dict[idx], df_sorted], ignore_index=True)
        if enable_print:
            print(f"{img_path_lst[idx]} {api_provider.upper()} Text Confusion Matrix:")
            ocr_confusion_matrix(actual_data[idx][0], detected_texts)
            print(f"{img_path_lst[idx]} {api_provider.upper()} Book Names Confusion Matrix:")
            text_group_confusion_matrix(actual_data[idx][1], ocr_book_names.split("\n"))
            print('=========================================================')
    return text_df_dict


def save_df_dict(img_path_lst: list, df_dict: dict):
    """
    Save dict of Dataframe for future use
    :param img_path_lst: list -- list of image paths
    :param df_dict: dict -- collection of text info dataframes
    """
    for pic_name, df in zip(img_path_lst, df_dict.values()):
        img_name = pic_name.split("/")[-1]
        csv_name = img_name.split(".")[0]
        df.to_csv('./data/' + csv_name + '_df.csv', index=False, header=True, encoding='utf-8')


def load_df_dict(img_path_lst: list) -> dict:
    """
    Load past text info dataframes
    :param img_path_lst: list -- list of image paths
    :return: df_dict: dict -- collection of text info dataframes
    """
    df_dict = {}
    for idx, pic_name in enumerate(img_path_lst):
        img_name = pic_name.split("/")[-1]
        csv_name = img_name.split(".")[0]
        temp_df = pd.read_csv('./data/' + csv_name + '_df.csv')
        eval_df = eval_object_columns(temp_df)
        df_dict[idx] = eval_df
    return df_dict

