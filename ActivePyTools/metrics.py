import difflib

from ActivePyTools import constants as c
from ActivePyTools.utils import normalize


def calculate_idx_match_percentage(actual_lst: list, predicted_lst: list) -> (float, float):
    """
    Calculate precision and recall from actual and predicted label lists.

    :param actual_lst: List of actual index.
    :param predicted_lst: List of predicted index.
    :return: A tuple containing precision and recall.
    """
    predicted_set = set(predicted_lst)
    actual_set = set(actual_lst)

    TP = len(predicted_set.intersection(actual_set))
    FP = len(predicted_set) - TP
    FN = len(actual_set) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return precision, recall


def calculate_text_match_percentage(actual_lst, predicted_lst: list) -> (float, float, list):
    """
    calculate precision and recall of two lists of sentences
    :param actual_lst: Human detected book and author name collection (Actual)
    :param predicted_lst: Software detected book and author name collection (Predicted)
    :return:
        precision: TP / (TP + FP)
        recall:    TP / (TP + FN)
        tp_list: list of TP items
    """
    # Normalize
    if isinstance(actual_lst, str):
        normalized_actual_lst = set(normalize(actual_lst).split())
    else:
        normalized_actual_lst = [normalize(s) for s in actual_lst]
    normalized_predicted_lst = [normalize(s) for s in predicted_lst]

    if len(normalized_actual_lst) == 0 or len(normalized_predicted_lst) == 0:  # Prevent division by zero
        return 0, 0, []

    true_positives = 0

    # Calculate true positives and false negatives using SequenceMatcher
    tp_lst = []
    fp_lst = []
    for predicted in normalized_predicted_lst:
        was_matched = False
        for actual in normalized_actual_lst:
            if difflib.SequenceMatcher(None, predicted, actual).ratio() >= c.recall_threshold:
                true_positives += 1
                tp_lst.append(actual)
                was_matched = True
                break
        if not was_matched:
            fp_lst.append(predicted)

    fn_lst = [item for item in normalized_actual_lst if item not in tp_lst]

    TP = len(tp_lst)
    FP = len(fp_lst)
    FN = len(fn_lst)

    # Calculate match percentage
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return precision, recall, tp_lst


def highest_match(actual_book_names: list, str_list: list) -> list:
    """
    match the highest matched pair
    :param actual_book_names: list -- actual dataset.
    :param str_list:          list -- predicted dataset.
    :return combined_list:    list of tuple -- each contains the highest match pair
        (ture_book_author, predicted_book_author, (recall, precision))
    """
    highest_match_percentages = [(-1, -1)] * len(actual_book_names)  # Start with -1 to indicate no match
    highest_match_str = [-1] * len(actual_book_names)
    names = [-1] * len(actual_book_names)
    str_list_used = [False] * len(str_list)

    # Compare each book/author name with each string list
    for i in range(len(actual_book_names)):

        book_author = actual_book_names[i]
        for j, string_list in enumerate(str_list):
            if str_list_used[j]:
                continue

            precision, recall, _ = calculate_text_match_percentage(book_author, string_list)
            if recall > highest_match_percentages[i][0]:
                highest_match_percentages[i] = (recall, precision)
                highest_match_str[i] = string_list
                names[i] = book_author

                if recall >= c.recall_threshold and precision >= c.precision_threshold:
                    str_list_used[j] = True

    highest_match_percentages = [percent for percent in highest_match_percentages if percent != -1]
    highest_match_str = [s for s in highest_match_str if s != -1]
    names = [n for n in names if n != -1]
    combined_list = list(zip(names, highest_match_str, highest_match_percentages))
    return combined_list


def ocr_confusion_matrix(actual: list, predicted: list) -> (list, list, int):
    """
    Provide a confusion matrix base on the provided lists of words
    :param actual:    actual dataset
    :param predicted: predicted dataset
    :return:
        fp_lst: list of FP
        fn_lst: list of FN
        TP: number of TP
    """
    matched_actual = []
    actual_copy = actual.copy()
    fp_lst = []
    fn_lst = []

    for p in predicted:
        match_found = False
        for a in actual:
            similarity = difflib.SequenceMatcher(None, normalize(a), normalize(p)).ratio()
            if similarity >= c.recall_threshold / 100:
                if a in actual_copy:
                    actual_copy.remove(a)
                    matched_actual.append(a)
                    match_found = True
                    break
                else:
                    continue

        if not match_found:
            fp_lst.append(p)
    TP = len(matched_actual)
    FP = len(fp_lst)

    # Check for False Negatives
    matched_actual_copy = matched_actual.copy()
    for a in actual:
        if a not in matched_actual_copy:
            fn_lst.append(a)
        else:
            matched_actual_copy.remove(a)
    FN = len(fn_lst)

    print(f"{'Matched by both':<30} {TP:<20}")
    print(f"{'Only by OCR':<30} {FP:<20}")
    print(f"{'OCR Text Total Amount':<30} {len(predicted):<20}")
    print(f"{'Only by Human':<30} {FN:<20}")
    print(f"{'Human Eye Book Name Total Amount':<30} {len(actual):<20}\n")

    print(f"{'OCR Text Recall':<30} {TP / len(actual):<20}")
    print(f"{'OCR Text Precision':<30} {TP / len(predicted):<20}")
    print()

    return fp_lst, fn_lst, TP


def update_text_group_confusion_matrix(actual_names: list, book_names: list) -> (int, int, int):
    """
    calculate TP, FP, FN base on the provided lists of sentences
    :param actual_names: list -- actual dataset
    :param book_names:   list -- predicted dataset
    :return: TP, FP, FN
    """
    hm_lst = highest_match(actual_names, book_names)
    temp_actual_names = actual_names.copy()
    temp_hm_lst = hm_lst.copy()
    temp_predicted_names = book_names.copy()
    # pprint.pprint(hm_lst)
    for hm in hm_lst:
        if hm[0] in temp_actual_names and hm[2][0] >= c.recall_threshold:  # and hm[2][1] >= c.precision_threshold:
            if hm[1] in temp_predicted_names:
                temp_actual_names.remove(hm[0])
                temp_predicted_names.remove(hm[1])
                temp_hm_lst.remove(hm)
    TP = len(actual_names) - len(temp_actual_names)
    FP = len(temp_predicted_names)
    FN = len(temp_actual_names)

    return TP, FP, FN


def text_group_confusion_matrix(actual_names: list, book_names: list, enable_print: bool = False) -> int:
    """
    Provide a confusion matrix base on the provided lists of sentences
    :param actual_names: list -- actual dataset
    :param book_names:   list -- predicted dataset
    :param enable_print: bool -- print or not
    :return: TP
    """
    if book_names == ['']:
        if enable_print:
            print("The OCR detected book names is none")
        return 0
    TP, FP, FN = update_text_group_confusion_matrix(actual_names, book_names)
    if enable_print:
        print_confusion_matrix(TP, FP, FN)

    return TP


def print_confusion_matrix(tp: int, fp: int, fn: int, img_name: None):
    if img_name is None:
        print(f"Book Name Group Larger than {c.recall_threshold}%:")
    else:
        print(f"{img_name} Book Name Group Larger than {c.recall_threshold}%:")
    print(f"{'Matched by both':<20} {tp:<20}")
    print(f"{'Only by OCR':<20} {fp:<20}")
    print(f"{'OCR Book Name Total Amount':<30} {tp + fp:<20}")
    print(f"{'Only by Human':<20} {fn:<20}")
    print(f"{'Human Eye Book Name Total Amount':<30} {tp + fn:<20}\n")

    print(f"{'OCR Book Name Recall':<30} {tp / (tp + fn):<20}")
    print(f"{'OCR Book Name Precision':<30} {tp / (tp + fp):<20}")
    print('=========================================================')
