from typing import Mapping, Any

import numpy as np
from pathlib import Path
import scipy.interpolate as interp
import scipy.optimize as sciopt

from bb_eval_engine.circuits.ngspice.netlist import NgSpiceWrapper, StateValue

class TwoStageOpenLoop(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:

        vout, freq, ibias = results['vout'], results['freq'], results['ibias']
        # Post process raw data
        gain = self.find_dc_gain(vout)
        ugbw = self.find_ugbw(freq, vout)
        phm = self.find_phm(freq, vout)

        return dict(ugbw=ugbw, gain=gain, phm=phm, Ibias=ibias)

    @classmethod
    def parse_output(cls, state) -> Mapping[str, np.ndarray]:

        ac_fname = Path(state['ac'])
        dc_fname = Path(state['dc'])

        if not ac_fname.is_file():
            print(f"ac file doesn't exist: {ac_fname}")
        if not dc_fname.is_file():
            print(f"ac file doesn't exist: {dc_fname}")

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag
        ibias = -dc_raw_outputs[1]

        return dict(freq=freq, vout=vout, ibias=ibias)

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]

    @classmethod
    def find_ugbw(cls, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = cls._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw
        else:
            return freq[0]

    @classmethod
    def find_phm(cls, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees

        # plt.subplot(211)
        # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # plt.subplot(212)
        # plt.plot(np.log10(freq[:200]), phase)

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = cls._get_best_crossing(freq, gain, val=1)
        if valid:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            return -180

    @classmethod
    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False


class TwoStageCommonModeGain(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:

        vout, freq = results['vout'], results['freq']
        # Post process raw data
        gain = self.find_dc_gain(vout)

        return dict(cm_gain=gain)

    @classmethod
    def parse_output(cls, state) -> Mapping[str, np.ndarray]:
        cm_fname = Path(state['cm'])

        if not cm_fname.is_file():
            print(f"cm file doesn't exist: {cm_fname}")

        cm_raw_outputs = np.genfromtxt(cm_fname, skip_header=1)
        freq = cm_raw_outputs[:, 0]
        vout_real = cm_raw_outputs[:, 1]
        vout_imag = cm_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return dict(freq=freq, vout=vout)

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]


class TwoStagePowerSupplyGain(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:

        vout, freq = results['vout'], results['freq']
        gain = self.find_dc_gain(vout)

        return dict(ps_gain=gain)

    @classmethod
    def parse_output(cls, state):
        ps_fname = Path(state['ps'])

        if not ps_fname.is_file():
            print(f"ps file doesn't exist: {ps_fname}")

        ps_raw_outputs = np.genfromtxt(ps_fname, skip_header=1)
        freq = ps_raw_outputs[:, 0]
        vout_real = ps_raw_outputs[:, 1]
        vout_imag = ps_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag

        return dict(freq=freq, vout=vout)

    @classmethod
    def find_dc_gain (cls, vout):
        return np.abs(vout)[0]

class TwoStageTransient(NgSpiceWrapper):

    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        return results


    @classmethod
    def parse_output(cls, state):
        tran_fname = Path(state['tran'])

        if not tran_fname.is_file():
            print(f"ac file doesn't exist: {tran_fname}")

        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        time =  tran_raw_outputs[:, 0]
        vout =  tran_raw_outputs[:, 1]
        vin =   tran_raw_outputs[:, 3]

        return dict(time=time, vout=vout, vin=vin)


    @classmethod
    def get_tset(cls, t, vout, vin, fbck, tot_err=0.1, plt=False):

        # since the evaluation of the raw data needs some of the constraints we need to do tset calculation here
        vin_norm = (vin-vin[0])/(vin[-1]-vin[0])
        ref_value = 1/fbck * vin
        y = (vout-vout[0])/(ref_value[-1]-ref_value[0])

        if plt:
            import matplotlib.pyplot as plt
            plt.plot(t, vin_norm/fbck)
            plt.plot(t, y)
            plt.figure()
            plt.plot(t, vout)
            plt.plot(t, vin)

        last_idx = np.where(y < 1.0 - tot_err)[0][-1]
        last_max_vec = np.where(y > 1.0 + tot_err)[0]
        if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
            last_idx = last_max_vec[-1]
            last_val = 1.0 + tot_err
        else:
            last_val = 1.0 - tot_err

        if last_idx == t.size - 1:
            return t[-1]
        f = interp.InterpolatedUnivariateSpline(t, y - last_val)
        t0 = t[last_idx]
        t1 = t[last_idx + 1]
        return sciopt.brentq(f, t0, t1)

