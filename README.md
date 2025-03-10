# My Project

**Forecasting of Product Sales Time Series in Scikit-Learn and PyTorch**

## Abstract

This thesis presents an extensive study on forecasting product sales using time series analysis in Scikit-Learn and PyTorch. The research focused on completely rewriting my bachelor's thesis and incorporating additional machine learning models to evaluate their performance compared to previous work. The results from the original thesis were used as a baseline for comparison. A key challenge in this study was the particularly short time series data (9 values as input, 3 to forecast) with no apparent regularity. This research aimed to develop the best models under these constraints, trying to outperform those in my thesis.

## Introduction

The primary goal of this study was to enhance the existing product sales forecasting methodology by employing various machine learning models. The models included:
- Naive
- Linear Regression
- Random Forest
- Fully Connected Network
- 1D Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Methodology

The training was conducted with and without outlier removal, utilizing Mean Squared Error (MSE) and Huber Loss as the loss functions. To ensure thorough evaluation, different model types were created with varying depths and corresponding hyperparameter tuning.

## Results

The performance of each model was assessed based on their ability to forecast sales accurately. Despite the short time series data, simpler models outperformed more complex ones, confirming that the choice of the model should align with the problem type and the available data.

## Conclusion

This study underscores the critical importance of selecting models for forecasting that align with the specific problem characteristics and data constraints. The findings suggest that simpler models can sometimes be more effective than more complex ones, particularly in cases involving short time series data with no apparent regularity. Future research could explore additional techniques and models to further improve forecasting accuracy under such constraints. These insights provide valuable guidance for practitioners and researchers in the field of time series analysis.

## How to Run the Project

1. Clone the repository:
   `git clone https://github.com/marco6872/forecasting-sales.git`
2. Create a virtual environment into the forecasting-sales directory and activate it:
   `python3 -m venv venv`
   `source venv/bin/activate`   # On Windows use `venv\Scripts\activate`
3. Install the dependencies:
   `pip install -r requirements.txt`
4. Install PyTorch:
   follow the official PyTorch installation guide ad `https://pytorch.org/`
   (Only Torch is necessary. Neither TorchVision nor Torchaudio are needed.)
5. Run the main.py file to start the project:
   `python3 main.py`

### Versions used

-  Python: 3.12
-  PyTorch: 2.6.0
   
