from typing import Sequence

from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.data.design import Design


class Flow(NgspiceFlowManager):

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):
        netlist_params = []
        for dsn in batch_of_designs:
            netlist_params.append(self.get_netlist_params(dsn, ['dc'], name='dc'))

        raw_results = self.ngspice_lut['dc'].run(netlist_params, verbose=self.verbose)
        results = [res[1] for res in raw_results]

        return results
