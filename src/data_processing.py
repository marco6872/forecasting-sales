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
    minmax_normalization=False,
    yeo_johnson=False):
    """
    Split the data into training and test sets.

    Parameters:
    df (DataFrame): The dataset.
    test_size (int): The number of samples in the test set.
    outlier_removal (bool): Whether to remove outliers from the training set.
    minmax_normalization (bool): Whether to apply Min-Max normalization.
    yeo_johnson (bool): Whether to apply Yeo-Johnson transformation.

    Returns:
    tuple: The training and test sets (X_train, X_test, y_train, y_test).
    """

    df = remove_unrelated_features(df, START_COLUMN, NUMBER_OF_COLUMNS_TO_KEEP)
    df = look_for_missing_data(df)
    df = negative_values_to_zero(df)

    train_data = df.iloc[:-test_size, :].values
    test_data = df.iloc[-test_size:, :].values

    if outlier_removal:
        train_data = remove_outliers(train_data)

    minmax_scaler = None
    if minmax_normalization:
        train_data, test_data, minmax_scaler = min_max_normalization(train_data, test_data)

    if yeo_johnson:
        pass


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


    # X_train, y_train = train_data[:, :in_len], train_data[:, in_len:]
    # X_test, y_test = test_data[:, :in_len], test_data[:, in_len:]

    return X_train, y_train, X_test, y_test, minmax_scaler


def original_data_statistics(X_train):
    """
    Perform statistical analysis on the original training dataset.

    Parameters:
    X_train (ndarray): The training dataset.

    Returns:
    None
    """
    df_train = pd.DataFrame(X_train, columns=[f'Feature {i+1}' for i in range(X_train.shape[1])])
    print(df_train.describe())
    print('\nNormality tests for original X_train data:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analysis_original_data.jpg')


def apply_yeo_johnson_transformation(X_train, X_test):   ############### DA SISTEMARE  ############################################
    """
    Apply Yeo-Johnson transformation to improve data distribution.

    Parameters:
    X_train (ndarray): The training dataset.
    X_test (ndarray): The test dataset.

    Returns:
    tuple: The transformed training, validation, and test sets (X_train, X_test).
    """
    pt = PowerTransformer(method='yeo-johnson')
    X_train = pt.fit_transform(X_train)
    X_test = pt.transform(X_test)
    
    # Perform normality tests and plot the results for X_train
    print('\nNormality tests for Yeo-Johnson transformed X_train data:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analysis_after_Yeo-Johnson_transformation.jpg')
    
    return X_train, X_test


def perform_normality_tests(data):
    """
    Perform normality tests on the data.

    Parameters:
    data (ndarray): The dataset.

    Returns:
    None
    """
    # Kolmogorov-Smirnov Test
    stat, p = stats.kstest(data.flatten(), 'norm')
    print('Kolmogorov-Smirnov Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Normal distribution (Kolmogorov-Smirnov)')
    else:
        print('Non-normal distribution (Kolmogorov-Smirnov)')
    
    # Anderson-Darling Test
    result = stats.anderson(data.flatten(), dist='norm')
    print('Anderson-Darling Statistics=%.3f' % result.statistic)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f'Normal distribution (Anderson-Darling) at significance level {sl}%')
        else:
            print(f'Non-normal distribution (Anderson-Darling) at significance level {sl}%')


def load_data(dataset_filename, test_size, remove_outliers=False, minmax_normalization=False, yeo_johnson=False):
    """
    Load preprocessed data or preprocess raw data based on user choice.

    Parameters:
    - preprocess (bool): Whether to preprocess the data or load preprocessed data.

    Returns:
    - tuple: X_train, X_test, y_train, y_test.
    """
    print(f'\nLoading raw data from "{dataset_filename}"...')
    df = import_dataset(dataset_filename)
    print(f'Raw dataframe:\n{df.head()}\n')

    X_train, y_train, X_test, y_test, minmax_scaler = preprocess_split_data(
        df,
        test_size=test_size,
        outlier_removal=remove_outliers,
        minmax_normalization=minmax_normalization,
        yeo_johnson=yeo_johnson,
    )

    # Save processed data
    def save_data(file_name, data):
        np.savetxt(PATH_TO_PROCESSED_DATA + file_name, data, delimiter=',')

    save_data('X_train.csv', X_train)
    save_data('X_test.csv', X_test)
    save_data('y_train.csv', y_train)
    save_data('y_test.csv', y_test)

    return X_train, y_train, X_test, y_test, minmax_scaler

