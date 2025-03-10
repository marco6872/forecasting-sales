# data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import scipy.stats as stats

from visualize import plot_feature_analysis


PATH_TO_PROCESSED_DATA = '../data/processed/'
PATH_TO_RAW_DATA = '../data/raw/'
START_COLUMN = 0
NUMBER_OF_COLUMNS_TO_KEEP = 12
NUMBER_OF_VALUES_TO_PREDICT = 3





def import_dataset(dataset_filename):
    """
    Import the dataset from a CSV file.

    Parameters:
    dataset_filename (str): The CSV file name.

    Returns:
    DataFrame: The imported dataset.
    """
    df = pd.read_csv(PATH_TO_RAW_DATA+dataset_filename)
    return df


def remove_unrelated_features(df, from_col, num_col_to_keep):
    """
    Remove features that are not related to the time series values.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with unrelated features removed.
    """
    df = df.iloc[:, from_col:num_col_to_keep]
    return df


def look_for_missing_data(df):
    """
    Look for missing data and remove rows with NaNs if any are found.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with rows containing NaNs removed.
    """
    missing_data = df.isnull().sum()
    print(f'\nMissing data:\n{missing_data}')
    if df.isnull().values.any():
        df = df.dropna()
        print("NaNs found and dropped.")
    else:
        print("No NaNs found.\n")
    return df


def negative_values_to_zero(df):
    """
    Look for negative data and convert them to zero if any are found.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with no nogative values.
    """

    # Count the number of values below 0 in each column
    count_below_zero_per_column = (df < 0).sum()
    print("Number of values below 0 per column:\n", count_below_zero_per_column)
    # Replace all negative numbers with zero
    df = df.clip(lower=0)
    print('Negative data converted to zero.\n')

    return df


def remove_outliers(data):
    """
    Remove rows with outliers in the dataset.

    Parameters:
    data (ndarray): The dataset.

    Returns:
    ndarray: The dataset with outliers removed.
    """
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    num_outliers = np.sum((data < lower_bound) | (data > upper_bound))

    print("\nNumber of outliers:", num_outliers)
    
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    
    data_filtered = data[mask]

    num_rows_removed = data.shape[0] - data_filtered.shape[0]
    print(f'Rows with outliers removed from dataset: {num_rows_removed}')
             
    return data_filtered


def min_max_normalization(train_data, test_data):
    """
    Apply Min-Max normalization to the datasets.

    Parameters:
    train_data (ndarray): The training dataset.
    test_data (ndarray): The test dataset.

    Returns:
    tuple: The normalized training and test sets (train_data, test_data).
    """
    print('Normalizing dataset...')
    train_flattened = train_data.flatten().reshape(-1, 1)
    test_flattened = test_data.flatten().reshape(-1, 1)

    scaler = MinMaxScaler()
    train_flattened = scaler.fit_transform(train_flattened)
    test_flattened = scaler.transform(test_flattened)

    train_scaled = train_flattened.reshape(train_data.shape)
    test_scaled = test_flattened.reshape(test_data.shape)

    print(f'Normalization factor  : {scaler.scale_}')
    print(f'Denormalization factor: {1/scaler.scale_}')
        
    return train_scaled, test_scaled, scaler


def preprocess_split_data(
    df,
    test_size,
    outlier_removal=False,
    minmax_normalization=True):
    """
    Split the data into training and test sets.

    Parameters:
    df (DataFrame): The dataset.
    test_size (int): The number of samples in the test set.
    outlier_removal (bool): Whether to remove outliers from the training set.
    minmax_normalization (bool): Whether to apply Min-Max normalization.

    Returns:
    tuple: The training and test sets (X_train, X_test, y_train, y_test).
    """

    class DummyScaler:
        def __init__(self, scale=1, minimum=0):
            self.scale_ = scale
            self.min_ = minimum

    df = remove_unrelated_features(df, START_COLUMN, NUMBER_OF_COLUMNS_TO_KEEP)
    df = look_for_missing_data(df)
    df = negative_values_to_zero(df)

    train_data = df.iloc[:-test_size, :].values
    test_data = df.iloc[-test_size:, :].values

    if outlier_removal:
        train_data = remove_outliers(train_data)

    if minmax_normalization:
        train_data, test_data, minmax_scaler = min_max_normalization(train_data, test_data)
    else:
        minmax_scaler = DummyScaler()

    in_len = NUMBER_OF_COLUMNS_TO_KEEP - NUMBER_OF_VALUES_TO_PREDICT
    out_len = NUMBER_OF_VALUES_TO_PREDICT

    def create_sliding_window(data, in_len, out_len):
        in_series, out_series = [], []
        for row in data:
            for i in range(out_len):
                split_idx = in_len + i
                in_segment, out_segment = row[i:split_idx], row[split_idx]
                in_series.append(in_segment)
                out_series.append(out_segment)
        return np.array(in_series), np.array(out_series)

    X_train, y_train = create_sliding_window(train_data, in_len, out_len)
    X_test, y_test = create_sliding_window(test_data, in_len, out_len)

    np.set_printoptions(precision=3, suppress=True)

    return X_train, y_train, X_test, y_test, minmax_scaler


def load_data(dataset_filename, test_size, remove_outliers=False, minmax_normalization=True):
    """
    Load preprocessed data or preprocess raw data based on user choice.

    Parameters:
    dataset_filename (str): The name of the CSV file containing the dataset.
    test_size (int): The number of samples to be used for the test set.
    remove_outliers (bool): Whether to remove outliers from the dataset. Default is False.
    minmax_normalization (bool): Whether to apply Min-Max normalization. Default is True.

    Returns:
    tuple: X_train, X_test, y_train, y_test, minmax_scaler
        X_train (np.array): Training features.
        y_train (np.array): Training target values.
        X_test (np.array): Test features.
        y_test (np.array): Test target values.
        minmax_scaler (object): Scaler object for inverse transforming the predictions.
    """
    print(f'\nLoading raw data from "{dataset_filename}"...')
    df = import_dataset(dataset_filename)
    print(f'Raw dataframe:\n{df.head()}\n')

    X_train, y_train, X_test, y_test, minmax_scaler = preprocess_split_data(
        df,
        test_size=test_size,
        outlier_removal=remove_outliers,
        minmax_normalization=minmax_normalization,
    )

    # Save processed data
    def save_data(file_name, data):
        np.savetxt(PATH_TO_PROCESSED_DATA + file_name, data, delimiter=',')

    save_data('X_train.csv', X_train)
    save_data('X_test.csv', X_test)
    save_data('y_train.csv', y_train)
    save_data('y_test.csv', y_test)

    return X_train, y_train, X_test, y_test, minmax_scaler

