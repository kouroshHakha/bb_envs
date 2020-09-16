from bb_eval_engine.util.importlib import import_bb_env

from utils.pdb import register_pdb_hook
register_pdb_hook()

import time
from pathlib import Path

from bb_envs.bmark_algs.tpe import Explorer

if __name__ == '__main__':
    env = import_bb_env('bb_envs/src/bb_envs/ngspice/envs/two_stage_opamp_1.yaml')
    seed =  10
    load = False
    fname = 'two_stage_env1_tpe_s10.pkl'

    s = time.time()
    explorer = Explorer(env, seed=seed)
    ffname = (Path('datasets') / fname)
    ffname.parent.mkdir(parents=True, exist_ok=True)
    if load:
        explorer.load(ffname)
    else:
        explorer.start(2000)
        explorer.save(ffname)

    breakpoint()