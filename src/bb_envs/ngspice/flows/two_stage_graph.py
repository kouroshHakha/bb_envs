"""
This is an example of using blackbox_eval_engine setup for a two_stage_opamp simulations.
It includes open-loop ac simulation with random location of input and transistor sizes.
"""

from typing import Mapping, Any, Sequence

import numpy as np

from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.data.design import Design

from ..wrappers.two_stage_graph import TwoStageOpenLoop


class TwoStageGraphFlow(NgspiceFlowManager):

    def __init__(self, *args, **kwargs):
        NgspiceFlowManager.__init__(self, *args, **kwargs)
        # self.fb_factor = kwargs['feedback_factor']
        # self.tot_err = kwargs['tot_err']

    # def interpret(self, design: Design, *args) -> Mapping[str, Any]:
    #     mode = args[0]

    #     if mode == 'ol':
    #         path_vars = ['ac', 'dc']
    #     elif mode == 'cm':
    #         path_vars = ['cm']
    #     elif mode == 'ps':
    #         path_vars = ['ps']
    #     elif mode == 'tran':
    #         path_vars = ['tran']
    #     else:
    #         raise ValueError('invalid mode!')

    #     design.specs.update(self.get_netlist_params(design, path_vars, name=mode))

    #     return design.specs

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):
        dsns = [self.get_netlist_params(dsn, ['ac', 'dc'], name='graph') for dsn in batch_of_designs]

        with self.ngspice_lut['graph'] as netlister:
            raw_results = netlister.run(dsns, verbose=self.verbose, debug=self.debug)
            results = [res[1] for res in raw_results]
    
        return results    
