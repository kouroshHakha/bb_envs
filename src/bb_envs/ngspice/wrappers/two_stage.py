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

        """
        NOTE:
            Input is assumed to be always ac=1 (real>0). It's also assumed that freq[0] is dc.
            In this case, real(gain[0]) < 0 means an inverting gain and real(gain[1]) > 0 means
            non-inverting.

            For non-inverting gain the origin of phase is 0 degrees (also returned by np.angle).
            so phase margin should be distance from -180 at fu.

            For inverting gain, the origin of phase should be 180 and the margin is defined as
            distance from 0 at fu. This is the case that should be post-processed further
            because np.angle does not always return the correct origin. Depending on circuit setup
            in some corner cases the origin could be inferred as -180.
            The solution is that in those cases phase is added to 360 to move it back to 180.

            for fu, gain is defined as abs(vout):
            1. if gain crosses 1 at some point, the first crossing is fu. In this case
            fstart < fu < fstop and pm can be determined. This is the typical case
            2. if gain is always below 1, it means that you are always stable
            (even with a terrible gain) so pm should be 180 (maximum)
            3. if gain is always greater than 1, there will be no crossing, but you will be stable
            if minimum phase margin is still large, so in this case it makes sense to return
            the minimum of phase margin across all observed frequencies.

            we will make pm non-negative (zero basically means you are not stable), so we will
            return max(pm, 0)

        """

        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw, valid = cls._get_best_crossing(freq, gain, val=1)

        gain_ltu = np.all(gain < 1)
        gain_gtu = np.all(gain > 1)
        # sanity check
        assert (gain_ltu or gain_gtu) is not valid, \
            ValueError(f'valid = {valid}, gain_ltu = {gain_ltu}, gain_gtu = {gain_gtu}')

        inv = np.real(vout[0]) < 0

        if valid:
            pm = phase_fun(ugbw)
        else:
            pm  = 180 if gain_ltu else np.min(phase)

        if inv and not (phase_fun(freq[0]) > 0):
            pm += 360
        elif not inv and not gain_ltu:
            pm += 180

        return max(pm, 0)

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

