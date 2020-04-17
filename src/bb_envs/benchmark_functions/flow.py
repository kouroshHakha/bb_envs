from typing import Sequence, Any, Callable, List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

    def plot_contours(self, ranges, ax=None, fpath='', show_fig=False):
        if ax is None:
            plt.close()
            ax = plt.axes()
        if len(ranges) != 2:
            raise ValueError('oops')

        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-5, 5, 100)
        x_mesh, y_mesh = np.meshgrid(x1, x2)
        xflat = x_mesh.flatten()
        yflat = y_mesh.flatten()

        xin = np.stack([xflat, yflat], axis=-1)
        yout = self.fn(xin)

        z_mesh = yout.reshape(x_mesh.shape)

        x_mesh, y_mesh = np.meshgrid(np.arange(100), np.arange(100))

        cp = ax.contour(x_mesh, y_mesh, z_mesh, np.min(yout) + np.arange(0, 5, 0.3), alpha=0.5)
        ax.clabel(cp, inline=True, fontsize=5)

        if fpath:
            fpath: Path = Path(fpath)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fpath, dpi=200)
        elif show_fig:
            plt.show()

class MultiOutputFnFlow:
    def __init__(self, fnames: Sequence[str]) -> None:

        self.fns: List[Callable] = []
        for fn in fnames:
            self.fns.append(registered_functions[fn])

    def batch_evaluate(self, batch_of_designs: Sequence[Design], **kwargs) -> Sequence[Any]:
        cur_order = list(batch_of_designs[0].value_dict.keys())
        params_vec = np.stack([list(x.value_dict.values()) for x in batch_of_designs], axis=0)
        desired_order = [f'x{i}' for i in range(params_vec.shape[-1])]
        desired_axis = [cur_order.index(var) for var in desired_order]

        correct_params_vec = params_vec[..., desired_axis]

        fvals = np.concatenate([fn(correct_params_vec).reshape(-1, 1) for fn in self.fns], axis=1)
        mults = np.prod(fvals, axis=-1)

        results = []
        for row, mult in zip(fvals, mults):
            result = dict(
                [(f'f{i}', fi) for i, fi in enumerate(row)] + [('valid', True), ('obj', mult)]
            )
            results.append(result)

        return results