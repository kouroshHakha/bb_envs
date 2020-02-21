from typing import Sequence, Any, Dict

import numpy as np
from dataclasses import dataclass
from math import log10

from bb_eval_engine.data.design import Design

@dataclass
class Transistor:
    type: str
    vth: float
    k: float
    lambda_: float
    cd_unit: float = 0
    cs_unit: float = 0


class CSAmpFlow:

    def __init__(self, sim: Dict[str, Any], *args, **kwargs) -> None:

        self.specs = sim
        self.tran = Transistor('nmos', **self.specs['nmos_model'])

    def get_gain(self, wl_ratio, vb, rload):
        # gain is zero for invalid designs (vb < vth or vds < vod)
        tran = self.tran
        vdd = self.specs['vdd']

        ibias = tran.k / 2 * wl_ratio * (vb - tran.vth) ** 2 * (vb >= tran.vth)
        vds = vdd - ibias * rload
        sat_cond = (vds >= (vb - tran.vth)) * (vb >= tran.vth)

        gm = tran.k * wl_ratio * (vb - tran.vth)
        ro = 1 / tran.lambda_ / (ibias + 1e-15)
        rl = rload * ro / (ro + rload)

        gain = gm * rl * sat_cond

        return gain, ibias


    def get_bw(self, wl_ratio, vb, rload):
        # bw is zero for invalid designs (vb < vth or vds < vod)
        tran = self.tran
        cload = self.specs.get('cload', 0)
        cr_unit = self.specs.get('cr_unit', 0)
        vdd = self.specs['vdd']

        ibias = tran.k / 2 * wl_ratio * (vb - tran.vth) ** 2 * (vb >= tran.vth)
        vds = vdd - ibias * rload
        sat_cond = (vds >= (vb - tran.vth)) * (vb >= tran.vth)

        ro = 1 / tran.lambda_ / (ibias + 1e-15)
        rl = rload * ro / (ro + rload)

        cpar = wl_ratio * tran.cd_unit + rl * cr_unit
        cl = cload + cpar

        bw = (1 / 2 / np.pi / rl / cl) * sat_cond

        return bw, ibias

    def compute (self, param_vec: np.ndarray):
        wl_ratio = param_vec[..., 0]
        vb = param_vec[..., 1]
        rload = param_vec[..., 2]
        av, _ = self.get_gain(wl_ratio, vb, rload)
        bw, ib = self.get_bw(wl_ratio, vb, rload)

        results = []

        def err(spec, target, plus=True):
            if plus:
                return (spec - target) / (spec + target + 1e-15)
            return (target - spec) / (spec + target + 1e-15)

        for av, bw, ib in zip(av, bw, ib):
            gain_err = err(av, 1.5)
            bw_err = err(bw, 1e9)
            ib_err = err(ib, 1e-3, False)
            res = dict(av=av,
                       bw=bw,
                       bw_log=log10(bw+1e-15),
                       ib=ib,
                       # valid=(bw!=0),
                       gbw=av*bw,
                       obj=max(-gain_err, -bw_err, -ib_err),
                       )
            results.append(res)
        return results

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs) -> Sequence[Any]:
        cur_order = list(batch_of_designs[0].value_dict.keys())
        params_vec = np.stack([list(x.value_dict.values()) for x in batch_of_designs], axis=0)
        desired_order = ['wl_ratio', 'vb', 'rload']
        desired_axis = [cur_order.index(var) for var in desired_order]

        correct_params_vec = params_vec[..., desired_axis]

        results = self.compute(correct_params_vec)

        return results


