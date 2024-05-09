import chardet
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from ActivePyTools.grab_data import eval_object_columns


loc_df_path = 'data/digit_data/loc_df.csv'
vec_df_path = 'data/digit_data/vec_df.csv'


def grab_df_data(df_path):
    with open(df_path, 'rb') as file:
        encoding = chardet.detect(file.read())['encoding']

    temp_df = pd.read_csv(df_path, encoding=encoding)
    df = eval_object_columns(temp_df)
    return df


def load_loc_data(root_path):
    loc_df = grab_df_data(root_path + loc_df_path)
    loc_df['label'] = loc_df['label'].astype(int)
    loc_df['idx'] = loc_df['idx'].astype(int)

    X_train = loc_df[loc_df['type'] == 'train'].drop(columns=['label', 'idx', 'type'])
    y_train = loc_df[loc_df['type'] == 'train']['label']
    X_valid = loc_df[loc_df['type'] == 'valid'].drop(columns=['label', 'idx', 'type'])
    y_valid = loc_df[loc_df['type'] == 'valid']['label']
    X_test = loc_df[loc_df['type'] == 'test'].drop(columns=['label', 'idx', 'type'])
    y_test = loc_df[loc_df['type'] == 'test']['label']

    return loc_df, X_train, y_train, X_valid, y_valid, X_test, y_test


def load_vec_data(root_path):
    vec_df = grab_df_data(root_path + vec_df_path)
    vec_df['label'] = vec_df['label'].astype(int)
    vec_df['idx'] = vec_df['idx'].astype(int)

    X_train = vec_df[vec_df['type'] == 'train'].drop(columns=['label', 'idx', 'type'])
    y_train = vec_df[vec_df['type'] == 'train']['label']
    X_valid = vec_df[vec_df['type'] == 'valid'].drop(columns=['label', 'idx', 'type'])
    y_valid = vec_df[vec_df['type'] == 'valid']['label']
    X_test = vec_df[vec_df['type'] == 'test'].drop(columns=['label', 'idx', 'type'])
    y_test = vec_df[vec_df['type'] == 'test']['label']

    return vec_df, X_train, y_train, X_valid, y_valid, X_test, y_test


def measure_performance(self, y, preds, enable_print=True):
    if self.preds is None:
        raise Exception("No Predictions Stored!")
    accuracy = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP)

    if enable_print:
        print(f"CM: \n{cm}")
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Specificity: {specificity:.2f}')
        print(f'Accuracy: {accuracy:.2f}')

    return precision, recall, specificity, cm
