from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from beras.core import Tensor
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
            Dense(784, 128, initializer="kaiming"),
            ReLU(),
            Dense(128, 64, initializer="kaiming"),
            ReLU(),
            Dense(64, 10, initializer="xavier"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    return Adam(learning_rate=0.001)

def get_loss_fn():
    return CategoricalCrossEntropy()

def get_acc_fn():
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    model.compile(get_optimizer(), get_loss_fn(), get_acc_fn())
    
    # 3. Load and preprocess the data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # One-hot encode labels
    ohe = OneHotEncoder()
    ohe.fit(y_train)
    y_train_oh = Tensor(ohe.forward(y_train))
    y_test_oh = Tensor(ohe.forward(y_test))
    
    # 4. Train the model
    history = model.fit(x_train, y_train_oh, epochs=8, batch_size=128)

    # 5. Evaluate the model
    metrics, predictions = model.evaluate(x_test, y_test_oh, batch_size=128)
    print("Test metrics:", metrics)

    np.save("predictions.npy", predictions)
