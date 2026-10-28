import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # mean of means: average over batch of the mean squared error per sample
        return Tensor(np.mean(np.mean((y_pred - y_true)**2, axis=-1)))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        batch_size = y_pred.shape[0]
        feature_size = y_pred.shape[-1]
        grad_pred = 2 * (y_pred - y_true) / (batch_size * feature_size)
        grad_true = np.zeros_like(y_true)  # not used
        return [Tensor(grad_pred), Tensor(grad_true)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return Tensor(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        batch_size = y_pred.shape[0]
        grad_pred = -y_true / y_pred / batch_size
        grad_true = np.zeros_like(y_true)
        return [Tensor(grad_pred), Tensor(grad_true)]
