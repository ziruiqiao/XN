from ActivePyTools import constants
from ActivePyTools.crop_functions import *
from analyze_book_name_functions import *
import re

import matplotlib.patches as patches
import boto3
import json


def ensure_string(value):
    if isinstance(value, str):
        return value
    else:
        return str(value)


def define_confusion_matrix(actual, predicted):
    TP = FP = FN = 0
    matched_actual = []
    actual_copy = actual.copy()
    fp_lst = []
    fn_lst = []

    print(f"Number of True Book Names: {len(actual)}")
    print(f"Number of Detected Book Name: {len(predicted)}")
    for p in predicted:
        match_found = False
        for a in actual:
            if ensure_string(a).lower() == ensure_string(p).lower():
                if a in actual_copy:
                    actual_copy.remove(a)
                    matched_actual.append(a)
                    TP += 1
                    match_found = True
                    break
                else:
                    continue

        if not match_found:
            FP += 1
            fp_lst.append(p)

    # Check for False Negatives
    matched_actual_copy = matched_actual.copy()
    for a in actual:
        if a not in matched_actual_copy:
            FN += 1
            fn_lst.append(a)
        else:
            matched_actual_copy.remove(a)


    print(f"{'Matched by both':<20} {TP:<20}")
    print(f"{'Only by OCR':<20} {FP:<20}")
    print(f"{'Only by Human':<20} {FN:<20}")
    print()

    return fp_lst, fn_lst


def collect_names(shelf: pd.DataFrame):
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


def get_rect(coord):
    return {
        'x': coord['Left'],
        'y': coord['Top'],
        'width': coord['Width'],
        'height': coord['Height']
    }


def get_text_data(text):
    coord = text.boundBox
    return get_rect(coord)


def draw_rectangle(rectangle, color='white'):
    rect = patches.Rectangle((rectangle['x'],
                              rectangle['y']),
                             rectangle['width'],
                             rectangle['height'],
                             linewidth=1,
                             edgecolor=color, facecolor='none')
    return rect


def plot_text_detected(im, df):
    if df is not None:
        all_text_rects = [draw_rectangle(get_text_data(annotation), color='blue')
                          for annotation in df.itertuples()]
    else:
        all_text_rects = []

    fix, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(im)

    for rect in all_text_rects:
        ax.add_patch(rect)

    plt.show()


def normalize(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower())


# Function to calculate match percentage
def calculate_match_percentage(book_author, string_list):
    # Normalize the book/author name
    normalized_book_author = set(normalize(book_author).split())
    # Normalize the strings in the string list
    normalized_string_list = [normalize(s) for s in string_list]

    # Calculate the number of matching words
    matches = sum(1 for word in normalized_string_list if word in normalized_book_author)
    detected_matches = sum(1 for word in normalized_book_author if word in normalized_string_list)

    # Calculate match percentage
    if len(normalized_book_author) == 0:  # Prevent division by zero
        return 0
    match_percentage1 = (matches / len(normalized_book_author)) * 100
    match_percentage2 = (detected_matches / len(normalized_string_list)) * 100
    return match_percentage1, match_percentage2


def highest_match(book_names, str_list):
    highest_match_percentages = [(-1, -1)] * len(book_names)  # Start with -1 to indicate no match
    highest_match_str = [-1] * len(book_names)
    names = [-1] * len(book_names)
    str_list_used = [False] * len(str_list)

    # Compare each book/author name with each string list
    for i in range(len(book_names)):

        book_author = book_names[i]
        for j, string_list in enumerate(str_list):
            if str_list_used[j]:
                continue

            match_percentage = calculate_match_percentage(book_author, string_list)
            if match_percentage[0] > highest_match_percentages[i][0]:
                highest_match_percentages[i] = match_percentage
                highest_match_str[i] = string_list
                names[i] = book_author

                if match_percentage[0] >= match_threshold and match_percentage[1] >= match_threshold:
                    str_list_used[j] = True

    highest_match_percentages = [percent for percent in highest_match_percentages if percent != -1]
    highest_match_str = [idx for idx in highest_match_str if idx != -1]
    names = [n for n in names if n != -1]
    combined_list = list(zip(names, highest_match_str, highest_match_percentages))
    return combined_list


def check_accuracy(img_paths, actual_names, cropped_pics, api, api_provider, useline):
    print('=========================================================')
    print(f'{api_provider.upper()} OCR with {"line" if useline else "word"}: \n')
    text_dict = {}
    for idx, (crops, ori_img) in enumerate(cropped_pics):
        path = img_paths[idx]
        img_name = path.split("/")[-1]
        text_dict[idx] = pd.DataFrame()

        predict_names = set()
        detected_texts = []
        total_true_match = 0
        total_false_neg = len(actual_names[idx][1])
        total_false_pos = 0
        total_positive = len(actual_names[idx][1])
        for idx2, lst in enumerate(crops):
            for idx3, image in enumerate(lst):
                cv2.imwrite(f'./temp_pics/{img_name}_{idx2}_{idx3}.jpg', image)
                book_names, text_df = collect_book_names(image, api, api_provider, useline)
                # predict_names.update(book_names)
                for value in text_df.itertuples():
                    detected_texts.append(value.txt)
                if len(text_df) > 0:
                    text_df['crop_idx'] = [(idx2, idx3)] * len(text_df)
                    text_df['Left'] = text_df['boundBox'].apply(lambda x: x['Left'])
                    text_df['Top'] = text_df['boundBox'].apply(lambda x: x['Top'])
                    df_sorted = text_df.sort_values(by=['Left', 'Top'])
                    text_dict[idx] = pd.concat([text_dict[idx], df_sorted], ignore_index=True)

                    hm_lst = highest_match(actual_names[idx][1], book_names)
                    temp_book_names = actual_names[idx][1].copy()
                    temp_hm_lst = hm_lst.copy()
                    temp_grouped_names = book_names.copy()
                    for hm in hm_lst:
                        print(hm)
                        if hm[0] in temp_book_names and hm[2][0] >= match_threshold and hm[2][1] >= match_threshold:
                            if hm[1] in temp_grouped_names:
                                temp_book_names.remove(hm[0])
                                temp_grouped_names.remove(hm[1])
                                temp_hm_lst.remove(hm)
                    true_match = len(actual_names[idx][1]) - len(temp_book_names)
                    actual_names[idx][1] = temp_book_names
                    false_positive = len(temp_grouped_names)
                    total_true_match += true_match
                    total_false_neg -= true_match
                    total_false_pos += false_positive

                # plot_text_detected(image, text_df)
        print(f"{img_paths[idx]} Detected Words Confusion Matrix:")
        fp1, fn1 = define_confusion_matrix(actual_names[idx][0], detected_texts)
        # print(f"Detected but not in Actual: {fp}\n")
        # print(f"In Actual but not Detected: {fn}")

        print(f"{img_paths[idx]} Book Name Group Larger than {match_threshold}%:")
        print(f"{'Matched by both':<20} {total_true_match:<20}")
        print(f"{'Only by OCR':<20} {total_false_pos:<20}")
        print(f"{'Only by Human':<20} {total_false_neg:<20}")
        # fp2, fn2 = define_confusion_matrix(actual_names[idx][1], list(predict_names))
        # print(f"Detected but not in Actual: {fp}\n")
        # print(f"In Actual but not Detected: {fn}")
        print('=========================================================')

    return text_dict


with open('../credentials.json') as f:
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

file_path1 = '../data/bookshelves_data.xlsx'
file_path2 = '../data/bookshelves_data_round2.xlsx'

all_sheets_df = pd.read_excel(file_path2, sheet_name=None)
# shelf_1 = all_sheets_df['Bookshelves_1']
# shelf_4 = all_sheets_df['Bookshelves_4']
# shelf_5 = all_sheets_df['Bookshelves_5']
shelf_6 = all_sheets_df['Bookshelves_6']
shelf_7 = all_sheets_df['Bookshelves_7']
shelf_8 = all_sheets_df['Bookshelves_8']
# actual_names_1_split, actual_names_1 = collect_names(shelf_1)
# actual_names_4_split, actual_names_4 = collect_names(shelf_4)
# actual_names_5_split, actual_names_5 = collect_names(shelf_5)
actual_names_6_split, actual_names_6 = collect_names(shelf_6)
actual_names_7_split, actual_names_7 = collect_names(shelf_7)
actual_names_8_split, actual_names_8 = collect_names(shelf_8)

img_paths = ['./pics/IMG_7940.jpeg', './pics/IMG_7941.jpeg', './pics/IMG_7942.jpeg']
actual_names = [[actual_names_6_split, actual_names_6], [actual_names_7_split, actual_names_7],
                [actual_names_8_split, actual_names_8]]
cropped_pics = []
for idx, path in enumerate(img_paths):
    im, cropped_img = crop_image(path)
    cropped_pics.append((cropped_img, im))
    # for idx2, lst in enumerate(cropped_img):
    #     for idx3, image in enumerate(lst):
    #         plot_text_detected(image, None)

print(constants.X_DISTANCE_THRESHOLD)
# check_accuracy(client, 'google', True)
txt_lst = check_accuracy(img_paths, actual_names, cropped_pics, client, 'google', False)
# check_accuracy(img_paths, actual_names, cropped_pics, rekognition, 'amazon', True)
# check_accuracy(img_paths, actual_names, cropped_pics, rekognition, 'amazon', False)

txt_lst[0].to_csv('pic6_df.csv', index=False, header=True, encoding='utf-8')
txt_lst[1].to_csv('pic7_df.csv', index=False, header=True, encoding='utf-8')
txt_lst[2].to_csv('pic8_df.csv', index=False, header=True, encoding='utf-8')

#%%
