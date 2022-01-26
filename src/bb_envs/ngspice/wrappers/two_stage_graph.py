from typing import Mapping, Any

import numpy as np
from pathlib import Path
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue

SRC_NAMES = set(['vvss', 'vvdd', 'vin1', 'vin2'])

class TwoStageOpenLoop(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        return results


    @classmethod
    def parse_output(cls, state) -> Mapping[str, np.ndarray]:

        ac_fname = Path(state['ac'])
        dc_fname = Path(state['dc'])

        if not ac_fname.is_file():
            print(f"ac file doesn't exist: {ac_fname}")
        if not dc_fname.is_file():
            print(f"ac file doesn't exist: {dc_fname}")

        nodes, vsrcs = cls._parse_ac(ac_fname)
        cls._update_ac_with_dc(dc_fname, nodes, vsrcs)

        return dict(nodes=nodes, vsrcs=vsrcs)

    @classmethod
    def _update_ac_with_dc(cls, fname, nodes, vsrc):
        raw_np = np.genfromtxt(fname, skip_header=1)
        headers = get_headers(fname)

        for col, header in zip(raw_np.T, headers):

            if '(' in header:
                name = cls._get_name(header)
                if name not in SRC_NAMES:
                    if name not in nodes:
                        raise ValueError(f'Could not find {header} in the ac results')
                    nodes[name]['dc'] = col
                else:
                    if name not in vsrc:
                        raise ValueError(f'Could not find {header} in the ac results')
                    vsrc[name]['dc'] = col
                
        return nodes, vsrc


    @classmethod
    def _parse_ac(cls, fname):

        raw_np = np.genfromtxt(fname, skip_header=1)
        headers = get_headers(fname)
        
        vsrc = {}
        nodes = {}
        attributes = {}
        rotation_idx = 0
        for col, header in zip(raw_np.T, headers):
            if rotation_idx % 3 == 0:
                # frequency
                attributes = {}
                attributes['freq'] = col
            elif rotation_idx % 3 == 1:
                attributes['real'] = col
            else:
                attributes['imag'] = col
                name = cls._get_name(header)
                if name not in SRC_NAMES:
                    nodes[name] = attributes
                else:
                    vsrc[name] = attributes
            rotation_idx += 1
        return nodes, vsrc

    @classmethod
    def _get_name(cls, header):
        name = header.split('(')[-1].split(')')[0]
        return name

def get_headers(txt_file):
    with open(txt_file) as f:
        headers = f.readline().split()
    return headers