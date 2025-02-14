import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_feature_analysis(data, filename=None, feature_names=None):
    """
    Plots histograms, Q-Q plots, and box plots for each feature in the data and saves the image.

    Parameters:
    - data: ndarray, shape (n_samples, n_features)
    - filename: str, optional, name of the file to save the image
    - feature_names: list of str, optional, names of the features
    """
    n_features = data.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=3, ncols=n_features, figsize=(20, 15))

    # Plot histograms, Q-Q plots, and box plots for each feature
    for i in range(n_features):
        # Histogram
        axes[0, i].hist(data[:, i], bins=30, edgecolor='k')
        axes[0, i].set_title(f'Histogram of {feature_names[i]}')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')

        # Q-Q Plot
        stats.probplot(data[:, i], dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot of {feature_names[i]}')

        # Box Plot
        axes[2, i].boxplot(data[:, i])
        axes[2, i].set_title(f'Box Plot of {feature_names[i]}')
        axes[2, i].set_xlabel('Feature')
        axes[2, i].set_ylabel('Value')

    # Adjust layout with more space around each graph
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    if filename:
        path = '../data/processed/'
        plt.savefig(path + filename)

    # Close the plot
    plt.close()

