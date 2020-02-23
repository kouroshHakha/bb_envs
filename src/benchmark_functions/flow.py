from typing import Sequence, Any, Callable

import numpy as np

from bb_eval_engine.data.design import Design
from .functions import registered_functions


class FnFlow:
    def __init__(self, fn: str) -> None:

        self.fn: Callable = registered_functions[fn]

    def batch_evaluate(self, batch_of_designs: Sequence[Design], **kwargs) -> Sequence[Any]:
        cur_order = list(batch_of_designs[0].value_dict.keys())
        params_vec = np.stack([list(x.value_dict.values()) for x in batch_of_designs], axis=0)
        desired_order = [f'x{i+1}' for i in range(params_vec.shape[-1])]
        desired_axis = [cur_order.index(var) for var in desired_order]

        correct_params_vec = params_vec[..., desired_axis]

        vals = self.fn(correct_params_vec)
        results = [{'val': val, 'valid': True} for val in vals]

        return results
