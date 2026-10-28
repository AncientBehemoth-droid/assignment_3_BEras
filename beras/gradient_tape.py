from collections import defaultdict
import numpy as np
from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        if target.ndim == 0:
            grads[id(target)] = [Tensor(np.array(1.0))]
        else:
            grads[id(target)] = [Tensor(np.ones_like(target))]

        visited = set()
        while queue:
            current = queue.pop(0)
            curr_id = id(current)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            
            layer = self.previous_layers[curr_id]
            if layer is None:
                continue
            
            upstream = grads[curr_id]
            
            if current.ndim == 0 and hasattr(layer, 'get_input_gradients'):
                input_grads = layer.get_input_gradients()
                for inp, ig in zip(layer.inputs, input_grads):
                    inp_id = id(inp)
                    grads[inp_id] = [ig]
                    if inp_id not in visited:
                        queue.append(inp)
                weight_grads = layer.get_weight_gradients()
                for w, wg in zip(layer.weights, weight_grads):
                    grads[id(w)] = [wg]
                continue
            
            weight_grads = layer.compose_weight_gradients(upstream)
            for w, wg in zip(layer.weights, weight_grads):
                w_id = id(w)
                if grads[w_id] is None:
                    grads[w_id] = [wg]
                else:
                    grads[w_id][0] = grads[w_id][0] + wg  
            
            input_grads = layer.compose_input_gradients(upstream)
            for inp, ig in zip(layer.inputs, input_grads):
                inp_id = id(inp)
                if grads[inp_id] is None:
                    grads[inp_id] = [ig]
                else:
                    grads[inp_id][0] = grads[inp_id][0] + ig
                if inp_id not in visited:
                    queue.append(inp)
        result = []
        for src in sources:
            g = grads[id(src)]
            if g is None:
                result.append(Tensor(np.zeros_like(src)))
            else:
                result.append(g[0] if isinstance(g, list) else g)
        return result
