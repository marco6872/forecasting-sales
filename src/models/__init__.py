# models/__init__.py

# Path for saving models
saved_models_path = './models/saved_models/'

# Import training and testing functions for different models
from .naive_model import train_and_test_naive_model
from .linearregression_model import train_and_test_linearregression_model
from .randomforest_model import train_and_test_randomforest_model
from .dl_linearregression_model import train_and_test_torch_linearregression_model
from .dl_fcn_model import train_and_test_fcn_model

# Dictionary mapping model names to their corresponding training and testing functions
model_dispatcher = {
    'naive': train_and_test_naive_model,
    'linearregression': train_and_test_linearregression_model,
    'randomforest': train_and_test_randomforest_model,
    'torch_linear_regression': train_and_test_torch_linearregression_model,
    'fully_connected_network': train_and_test_fcn_model,
}
