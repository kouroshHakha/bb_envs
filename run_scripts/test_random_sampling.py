from bb_eval_engine.util.importlib import import_bb_env
from utils.data.database import Database
from utils.file import write_pickle

from utils.pdb import register_pdb_hook
register_pdb_hook()

import time
from pathlib import Path

if __name__ == '__main__':
    env = import_bb_env('bb_envs/src/bb_envs/ngspice/envs/two_stage_opamp_2.yaml')
    fname = 'two_stage_random2.pkl'
    seed =  10

    ffname = (Path('datasets') / fname)
    s = time.time()
    db = Database(keep_sorted_list_of=['cost'])
    dsn_list = env.generate_rand_designs(2000, evaluate=True, seed=seed)
    db.extend(dsn_list)
    write_pickle(ffname, dict(db=db.picklable()))

    breakpoint()