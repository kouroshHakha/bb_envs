from typing import Mapping

from pathlib import Path
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue


class CsAmpNgspiceWrapper(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, StateValue]:

        # use parse output here
        freq, vout, ibias = results['freq'], results['vout'], results['ibias']
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)

        spec = dict(
            bw=bw,
            gain=gain,
            ibias=ibias
        )

        return spec

    @classmethod
    def parse_output(cls, state):

        ac_fname = Path(state['ac'])
        dc_fname = Path(state['dc'])

        if not ac_fname.is_file():
            print(f"ac file doesn't exist: {ac_fname}")
        if not dc_fname.is_file():
            print(f"ac file doesn't exist: {dc_fname}")

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout = ac_raw_outputs[:, 1]
        ibias = - dc_raw_outputs[1]

        return freq, vout, ibias

    @classmethod
    def find_dc_gain(cls, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_bw(cls, vout, freq):
        gain = np.abs(vout)
        gain_3db = gain[0] / np.sqrt(2)
        return cls._get_best_crossing(freq, gain, gain_3db)

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop
