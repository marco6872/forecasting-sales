import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from visualize import plot_feature_analysis

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

def remove_unrelated_features(df):
    """
    Remove features that are not related to the time series values.

    Parameters:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with unrelated features removed.
    """
    df = df.iloc[1:, 1:13]
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
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=0.3, random_state=0, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def perform_statistical_analysis(X_train):
    """
    Perform statistical analysis on the training dataset.

    Parameters:
    X_train (ndarray): The training dataset.

    Returns:
    None
    """
    df_train = pd.DataFrame(X_train, columns=[f'Feature {i+1}' for i in range(X_train.shape[1])])
    print(df_train.describe())
    print('\nNormality tests for original X_train data:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analisys_original_data.jpg', show_me=False)

def apply_yeo_johnson_transformation(X_train, X_val, X_test):
    """
    Apply Yeo-Johnson transformation to improve data distribution.

    Parameters:
    X_train (ndarray): The training dataset.
    X_val (ndarray): The validation dataset.
    X_test (ndarray): The test dataset.

    Returns:
    tuple: The transformed training, validation, and test sets (X_train, X_val, X_test).
    """
    pt = PowerTransformer(method='yeo-johnson')
    X_train = pt.fit_transform(X_train)
    X_val = pt.transform(X_val)
    X_test = pt.transform(X_test)
    print('\nNormality tests for Yeo-Johnson transformed X_train data:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analisys_after_Yeo-Johnson_transformation.jpg', show_me=False)
    return X_train, X_val, X_test

def analyze_and_cap_outliers(X_train):
    """
    Analyze and cap outliers in the training dataset.

    Parameters:
    X_train (ndarray): The training dataset.

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
    print('\nNormality tests for X_train data after outliers removal:\n')
    perform_normality_tests(X_train)
    plot_feature_analysis(X_train, 'feature_analisys_after_outliers_removal.jpg', show_me=False)
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
