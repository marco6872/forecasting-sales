# models/__init__.py

# Path for saving models
saved_models_path = './models/saved_models/'

# Import training and testing functions for different models
from .naive_model import train_and_test_naive_model
from .linearregression_model import train_and_test_linearregression_model
from .randomforest_model import train_and_test_randomforest_model
from .dl_linearregression_model import train_and_test_torch_linearregression_model
from .dl_fcn_model import train_and_test_fcn_model
from .dl_cnn_model import train_and_test_cnn_model
from .dl_lstm_model import train_and_test_lstm_model
from .dl_gru_model import train_and_test_gru_model

# Dictionary mapping model names to their corresponding training and testing functions
model_dispatcher = {
    'naive': train_and_test_naive_model,
    'linear regression': train_and_test_linearregression_model,
    'random forest': train_and_test_randomforest_model,
    'torch linear regression': train_and_test_torch_linearregression_model,
    'fully connected network': train_and_test_fcn_model,
    'convolutional neural network': train_and_test_cnn_model,
    'long short term memory': train_and_test_lstm_model,
    'gated recurrent unit': train_and_test_gru_model,
}
