import numpy as np

from preprocessing import (
    import_dataset,
    remove_unrelated_features,
    look_for_missing_data,
    split_data,
    perform_statistical_analysis,
    apply_yeo_johnson_transformation,
    analyze_and_cap_outliers,
    apply_min_max_normalization
)



###  DATA PREPROCESSING  ###################################################################


# Import the dataset
df = import_dataset('../data/raw/st_dataset.csv')

# Remove features that are not related to the time series values
df = remove_unrelated_features(df)

# Look for missing data
df = look_for_missing_data(df)

# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

# Perform statistical analysis on the training dataset
perform_statistical_analysis(X_train)

# Apply Yeo-Johnson transformation to improve data distribution
X_train, X_val, X_test = apply_yeo_johnson_transformation(X_train, X_val, X_test)

# Analyze and cap outliers in the training dataset
X_train = analyze_and_cap_outliers(X_train)

# Apply Min-Max normalization
X_train, X_val, X_test = apply_min_max_normalization(X_train, X_val, X_test)

# Save the data arrays as CSV files
path = '../data/processed/'
np.savetxt(path + 'X_train.csv', X_train, delimiter=',')
np.savetxt(path + 'X_val.csv', X_val, delimiter=',')
np.savetxt(path + 'X_test.csv', X_test, delimiter=',')
np.savetxt(path + 'y_train.csv', y_train, delimiter=',')
np.savetxt(path + 'y_val.csv', y_val, delimiter=',')
np.savetxt(path + 'y_test.csv', y_test, delimiter=',')









