import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import scipy.stats as stats

from visualize import plot_feature_analysis


PATH = '../data/processed/'
DATASET = '../data/raw/weekly_sales_dataset.csv'
START_COLUMN = 0
NUMBER_OF_COLUMNS_TO_KEEP = 12
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2



def import_dataset(filepath):
    """
    Import the dataset from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    DataFrame: The imported dataset.
    """
    df = pd.read_csv(filepath, header=None)
    return df


def remove_unrelated_features(df, from_col, num_col_to_keep):
    """
    Remove features that are not related to the time series values.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with unrelated features removed.
    """
    df = df.iloc[1:, from_col:num_col_to_keep]
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


def split_data(df):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    tuple: The training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X, y = df.iloc[:, :9].values, df.iloc[:, 9:].values
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=VALIDATION_SIZE, random_state=0, shuffle=True)

    # print number of lines for each dataset
    print('Original dataset shape:')
    print(f'  columns: {df.shape[1]} - rows: {df.shape[0]}\n')

    print('Train dataset shape (input):')
    print(f'  columns: {X_train.shape[1]} - rows: {X_train.shape[0]}\n')
    print('Validation dataset shape (input):')
    print(f'  columns: {X_val.shape[1]} - rows: {X_val.shape[0]}\n')
    print('Test dataset shape (input):')
    print(f'  columns: {X_test.shape[1]} - rows: {X_test.shape[0]}\n')

    return X_train, X_val, X_test, y_train, y_val, y_test


def original_data_statistics(X_train, show_plots):
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
    plot_feature_analysis(X_train, 'feature_analysis_original_data.jpg', show_me=show_plots)


def apply_yeo_johnson_transformation(X_train, X_val, X_test, show_plots):
    """
    Apply Yeo-Johnson transformation to improve data distribution.

    Parameters:
    X_train (ndarray): The training dataset.
    X_val (ndarray): The validation dataset.
    X_test (ndarray): The test dataset.
    show_plots (bool): Whether to show plots during preprocessing.

    Returns:
    tuple: The transformed training, validation, and test sets (X_train, X_val, X_test).
    """
    pt = PowerTransformer(method='yeo-johnson')
    X_train = pt.fit_transform(X_train)
    X_val = pt.transform(X_val)
    X_test = pt.transform(X_test)
    
    # Perform normality tests and plot the results for X_train
    print('\nNormality tests for Yeo-Johnson transformed X_train data:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analysis_after_Yeo-Johnson_transformation.jpg', show_me=show_plots)
    
    return X_train, X_val, X_test


def analyze_and_cap_outliers(X_train, show_plots):
    """
    Analyze and cap outliers in the training dataset.

    Parameters:
    X_train (ndarray): The training dataset.
    show_plots (bool): Whether to show plots during preprocessing.

    Returns:
    ndarray: The training dataset with outliers capped.
    """
    Q1 = np.percentile(X_train, 25, axis=0)
    Q3 = np.percentile(X_train, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    num_outliers = np.sum((X_train < lower_bound) | (X_train > upper_bound))
    
    print("\nNumber of outliers:", num_outliers)
    
    X_train = np.where(X_train < lower_bound, lower_bound, X_train)
    X_train = np.where(X_train > upper_bound, upper_bound, X_train)
    
    # Perform normality tests and plot the results for X_train
    print('\nNormality tests for X_train data after outliers removal:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analysis_after_outliers_removal.jpg', show_me=show_plots)
    
    return X_train


def apply_min_max_normalization(X_train, X_val, X_test):
    """
    Apply Min-Max normalization to the datasets.

    Parameters:
    X_train (ndarray): The training dataset.
    X_val (ndarray): The validation dataset.
    X_test (ndarray): The test dataset.

    Returns:
    tuple: The normalized training, validation, and test sets (X_train, X_val, X_test).
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test


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


def preprocess_data(X_train, X_val, X_test, show_plots=False):
    """
    Preprocess the data by applying selected transformations.

    Parameters:
    - X_train (ndarray): Training input data.
    - X_val (ndarray): Validation input data.
    - X_test (ndarray): Test input data.
    - show_plots (bool): Whether to show plots during preprocessing.

    Returns:
    - tuple: Preprocessed X_train, X_val, X_test.
    """
    # Apply transformations
    print('\n  Yeo-Johnson transformation')
    X_train, X_val, X_test = apply_yeo_johnson_transformation(X_train, X_val, X_test, show_plots)
    print('\n  Removing outliers')
    X_train = analyze_and_cap_outliers(X_train, show_plots)
    print('\n  Min-Max normalization')
    X_train, X_val, X_test = apply_min_max_normalization(X_train, X_val, X_test)

    return X_train, X_val, X_test
    

def load_data(preprocess=False, show_plots=False):
    """
    Load preprocessed data or preprocess raw data based on user choice.

    Parameters:
    - preprocess (bool): Whether to preprocess the data or load preprocessed data.
    - show_plots (bool): Whether to show plots during preprocessing.

    Returns:
    - tuple: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    print('\nLoading raw data...')
    df = import_dataset(DATASET)
    df = remove_unrelated_features(df, START_COLUMN, NUMBER_OF_COLUMNS_TO_KEEP)
    df = look_for_missing_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    original_data_statistics(X_train, show_plots)

    if preprocess:
        print("\nPreprocessing data...")
        X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test, show_plots)

    # Save processed data
    np.savetxt(PATH + 'X_train.csv', X_train, delimiter=',')
    np.savetxt(PATH + 'X_val.csv', X_val, delimiter=',')
    np.savetxt(PATH + 'X_test.csv', X_test, delimiter=',')
    np.savetxt(PATH + 'y_train.csv', y_train, delimiter=',')
    np.savetxt(PATH + 'y_val.csv', y_val, delimiter=',')
    np.savetxt(PATH + 'y_test.csv', y_test, delimiter=',')

    return X_train, X_val, X_test, y_train, y_val, y_test










        
    # else:
    #     print("Loading preprocessed data...")
    #     X_train = np.loadtxt(PATH + 'X_train.csv', delimiter=',')
    #     X_val = np.loadtxt(PATH + 'X_val.csv', delimiter=',')
    #     X_test = np.loadtxt(PATH + 'X_test.csv', delimiter=',')
    #     y_train = np.loadtxt(PATH + 'y_train.csv', delimiter=',')
    #     y_val = np.loadtxt(PATH + 'y_val.csv', delimiter=',')
    #     y_test = np.loadtxt(PATH + 'y_test.csv', delimiter=',')
