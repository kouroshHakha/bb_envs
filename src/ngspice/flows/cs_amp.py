from typing import Mapping, Any, Sequence

from bb_eval_engine.circuits.ngspice.flow import NgspiceFlowManager
from bb_eval_engine.util.design import Design


class CSAmpFlow(NgspiceFlowManager):

    def interpret(self, design: Design, *args, **kwargs) -> Mapping[str, Any]:
        params_dict = design['value_dict']
        params_dict.update(self.update_netlist_model_paths(design, ['ac', 'dc'], name='ac_dc'))

        return params_dict

    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs):
        interpreted_designs = [self.interpret(design) for design in batch_of_designs]
        raw_results = self.ngspice_lut['ac_dc'].run(interpreted_designs, verbose=self.verbose)
        results = [res[1] for res in raw_results]

        return results