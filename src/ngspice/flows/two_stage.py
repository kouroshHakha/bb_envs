"""
This is an example of using blackbox_eval_engine setup for a two_stage_opamp simulations.
It includes open-loop ac, transient, power supply rejection and common mode testbenches.
"""

from typing import Mapping, Any, Sequence, Dict

import numpy as np
from pathlib import Path

from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.data.design import Design

from ..wrappers.two_stage import TwoStageTransient

class TwoStageFlow(NgspiceFlowManager):

    def __init__(self, *args, **kwargs):
        NgspiceFlowManager.__init__(self, *args, **kwargs)
        self.fb_factor = kwargs['feedback_factor']
        self.tot_err = kwargs['tot_err']

    def interpret(self, design: Design, *args, **kwargs) -> Mapping[str, Any]:

        mode = args[0]

        if mode == 'ol':
            path_vars = ['ac', 'dc']
        elif mode == 'cm':
            path_vars = ['cm']
        elif mode == 'ps':
            path_vars = ['ps']
        elif mode == 'tran':
            path_vars = ['tran']
        else:
            raise ValueError('invalid mode!')

        design.specs.update(self.get_netlist_params(design, path_vars, name=mode))

        return design.specs

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):
        ol_dsns, cm_dsns, ps_dsns, tran_dsns = [], [], [], []
        for dsn in batch_of_designs:
            ol_dsns.append(self.get_netlist_params(dsn, ['ac', 'dc'], name='ol'))
            cm_dsns.append(self.get_netlist_params(dsn, ['cm'], name='cm'))
            ps_dsns.append(self.get_netlist_params(dsn, ['ps'], name='ps'))
            tran_dsns.append(self.get_netlist_params(dsn, ['tran'], name='tran'))

        results_tbs = []
        for tb, dsns in zip(('ol', 'cm', 'ps', 'tran'), (ol_dsns, cm_dsns, ps_dsns, tran_dsns)):
            with self.ngspice_lut[tb] as netlister:
                raw_results = netlister.run(dsns, verbose=self.verbose)
                results_tbs.append([res[1] for res in raw_results])

        # raw_results = self.ngspice_lut['ol'].run(ol_dsns, verbose=self.verbose)
        # results_ol = [res[1] for res in raw_results]
        # raw_results = self.ngspice_lut['cm'].run(cm_dsns, verbose=self.verbose)
        # results_cm = [res[1] for res in raw_results]
        # raw_results = self.ngspice_lut['ps'].run(ps_dsns, verbose=self.verbose)
        # results_ps = [res[1] for res in raw_results]
        # raw_results = self.ngspice_lut['tran'].run(tran_dsns, verbose=self.verbose)
        # results_tran = [res[1] for res in raw_results]

        results_eval = []
        for ol, cm, ps, tran in zip(*results_tbs):
            results_eval.append(self._get_specs(ol, cm, ps, tran))

        return results_eval

    def _get_specs(self, result_ol, result_cm, result_ps, result_tran):
        fdbck = self.fb_factor
        tot_err = self.tot_err

        ugbw_cur = result_ol['ugbw']
        gain_cur = result_ol['gain']
        phm_cur = result_ol['phm']
        ibias_cur = result_ol['Ibias']

        # common mode gain and cmrr
        cm_gain_cur = result_cm['cm_gain']
        cmrr_cur = 20 * np.log10(gain_cur / cm_gain_cur)  # in db
        # power supply gain and psrr
        ps_gain_cur = result_ps['ps_gain']
        psrr_cur = 20 * np.log10(gain_cur / ps_gain_cur)  # in db

        # transient settling time and offset calculation
        t = result_tran['time']
        vout = result_tran['vout']
        vin = result_tran['vin']

        tset_cur = TwoStageTransient.get_tset(t, vout, vin, fdbck, tot_err=tot_err)
        offset_curr = abs(vout[0] - vin[0] / fdbck)

        specs_dict = dict(
            gain=gain_cur,
            ugbw=ugbw_cur,
            pm=phm_cur,
            ibias=ibias_cur,
            cmrr=cmrr_cur,
            psrr=psrr_cur,
            offset_sys=offset_curr,
            tset=tset_cur,
        )

        return specs_dict