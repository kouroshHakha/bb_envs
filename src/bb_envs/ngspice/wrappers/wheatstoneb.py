from typing import Mapping, Any

import numpy as np
from pathlib import Path
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue

SRC_NAMES = set(['1', '6'])

class Wrapper(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        return results


    @classmethod
    def parse_output(cls, state) -> Mapping[str, np.ndarray]:

        dc_fname = Path(state['dc'])

        if not dc_fname.is_file():
            print(f"ac file doesn't exist: {dc_fname}")

        nodes, vsrcs = cls._get_dc(dc_fname)
        return dict(nodes=nodes, vsrcs=vsrcs)

    @classmethod
    def _get_dc(cls, fname):
        raw_np = np.genfromtxt(fname, skip_header=1)
        headers = get_headers(fname)

        vsrc = {}
        nodes = {}

        for col, header in zip(raw_np.T, headers):

            if '(' in header:
                name = cls._get_name(header)
                if name not in SRC_NAMES:
                    nodes[name] = {'dc': float(col)}
                else:
                    vsrc[name] = {'dc': float(col)}
                
        return nodes, vsrc

    @classmethod
    def _get_name(cls, header):
        name = header.split('(')[-1].split(')')[0]
        return name

def get_headers(txt_file):
    with open(txt_file) as f:
        headers = f.readline().split()
    return headers