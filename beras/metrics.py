import numpy as np
from beras.core import Tensor
from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        pred_classes = np.argmax(probs, axis=-1)
        true_classes = np.argmax(labels, axis=-1)
        return Tensor(np.mean(pred_classes == true_classes))
