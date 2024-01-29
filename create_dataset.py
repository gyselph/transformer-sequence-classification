"""
Make sure to download the dataset before running this script.
Download the dataset from here: https://www.kaggle.com/c/career-con-2019/data
Store the dataset here: "./career-con-2019"
"""


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess(x_raw, y_raw):
    # normalize features of x_raw
    column_names = x_raw.columns
    x_transformer = ColumnTransformer(
        [
           ("numeric", StandardScaler(), column_names[3:]),
        ],
        remainder='passthrough', n_jobs=-1)
    normalized = x_transformer.fit_transform(x_raw)
    new_columns = x_transformer.get_feature_names_out()
    x_preprocessed = pd.DataFrame(normalized, columns = new_columns)

    # discard unneseccary features of x and convert from 2d dataframe to 3d numpy array
    num_sequences = max(x_preprocessed["remainder__series_id"]) + 1
    num_time_steps = max(x_preprocessed["remainder__measurement_number"]) + 1
    num_features = 10
    x = np.zeros((num_sequences, num_time_steps, num_features))
    for i in range(len(x_preprocessed)):
        series_id = x_preprocessed.iloc[i]["remainder__series_id"]
        measurement_number = x_preprocessed.iloc[i]["remainder__measurement_number"]
        data = list(x_preprocessed.iloc[i])[:num_features]
        x[series_id][measurement_number] = data
    x_feature_names = new_columns[:num_features]

    # one-hot encode y_raw
    y_transformer = OneHotEncoder()
    y_raw = np.array(y_raw["surface"]).reshape(-1,1)
    y_transformer.fit(y_raw)
    y_category_names = y_transformer.categories_[0]
    y_preprocessed = y_transformer.transform(y_raw).toarray()

    return (x, y_preprocessed, x_feature_names, y_category_names)


def create_dataset(x_input_file, y_input_file, dataset_file):
    x_raw = pd.read_csv(x_input_file)
    y_raw = pd.read_csv(y_input_file)
    (x_preprocessed, y_preprocessed, x_feature_names, y_category_names) = preprocess(x_raw, y_raw)
    x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y_preprocessed, test_size=0.4)
    np.savez_compressed(dataset_file, x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, x_feature_names = x_feature_names, y_category_names = y_category_names)


if __name__ == "__main__":
    x_input_file = "career-con-2019/X_train.csv"
    y_input_file = "career-con-2019/y_train.csv"
    dataset_file = "dataset_career_con_2019.npz"
    create_dataset(x_input_file, y_input_file, dataset_file)