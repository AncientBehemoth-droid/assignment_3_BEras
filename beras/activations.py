import numpy as np
from typing import List

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        return Tensor(np.where(x > 0, x, self.alpha * x))

    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        x = self.inputs[0]
        grad = np.where(x > 0, 1.0, self.alpha)
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)

class GELU(Activation):
    """GELU using the tanh approximation.

      Equivalent to torch.nn.GELU(approximate="tanh").
      Reference: https://arxiv.org/abs/1606.08415
      """
    ## TODO: Implement GELU for intermediate activation. Hint: Use the tanh approximation for GELU, which is faster to compute than the exact formula.
    
    def forward(self, x: Tensor) -> Tensor:
        coeff = np.sqrt(2 / np.pi)
        inner = coeff * (x + 0.044715 * x**3)
        return Tensor(0.5 * x * (1 + np.tanh(inner)))

    def get_input_gradients(self) -> List[Tensor]:
        x = self.inputs[0]
        coeff = np.sqrt(2 / np.pi)
        u = coeff * (x + 0.044715 * x**3)
        tanh_u = np.tanh(u)
        du_dx = coeff * (1 + 3 * 0.044715 * x**2)
        sech2 = 1 - tanh_u**2
        grad = 0.5 * (1 + tanh_u) + 0.5 * x * sech2 * du_dx
        return [Tensor(grad)]

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x) -> Tensor:
        # Stable sigmoid
        return Tensor(1 / (1 + np.exp(-np.clip(x, -500, 500))))

    def get_input_gradients(self) -> list[Tensor]:
        y = self.outputs[0] if isinstance(self.outputs, list) else self.outputs
        y = self.forward(self.inputs[0])
        return [Tensor(y * (1 - y))]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def forward(self, x):
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        x_stable = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(x_stable)
        return Tensor(exps / np.sum(exps, axis=-1, keepdims=True))

    def get_input_gradients(self):
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        for i in range(bn):
            yi = y[i]
            grad[i] = np.diag(yi) - np.outer(yi, yi)
        
        return [Tensor(grad)]
        
        
