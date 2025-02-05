# models/__init__.py

saved_models_path = './models/saved_models/'

from .naive_model import train_naive_model, test_naive_model
from .linearregression_model import train_linearregression_model, test_linearregression_model
from .randomforest_model import train_randomforest_model, test_randomforest_model


model_dispatcher = {
    'naive': (train_naive_model, test_naive_model),
    'linearregression': (train_linearregression_model, test_linearregression_model),
    'randomforest': (train_randomforest_model, test_randomforest_model),
    
}


