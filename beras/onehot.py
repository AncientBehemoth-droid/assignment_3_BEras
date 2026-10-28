import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.classes_ = np.unique(data)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        self.n_classes = len(self.classes_)

    def forward(self, data):
        if not hasattr(self, 'class_to_idx'):
            self.fit(data)
        onehot = np.zeros((len(data), self.n_classes))
        for i, label in enumerate(data):
            if label in self.class_to_idx:
                onehot[i, self.class_to_idx[label]] = 1
        return onehot

    def inverse(self, data):
        indices = np.argmax(data, axis=-1)
        return self.classes_[indices]
