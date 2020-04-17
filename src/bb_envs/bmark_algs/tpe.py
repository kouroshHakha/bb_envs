"""This module is an optimizer that collects data from a cheap environement towards the solution"""

from bb_eval_engine.base import EvaluationEngineBase
from bb_eval_engine.data.design import Design

import numpy as np
from pathlib import Path
from hyperopt import hp, fmin, tpe, space_eval, Trials
from utils.data.database import Database
from utils.file import write_pickle, read_pickle

class Explorer:

    def __init__(self,
                 env: EvaluationEngineBase,
                 seed=10,
                 ):

        self.env = env
        self.seed = seed

        self.space = [hp.uniform(label, env.params_min[i], env.params_max[i]) for i, label in enumerate(env.design_vars)]

        self.db: Database[Design] = Database(keep_sorted_list_of=['cost'])
        self.trials = Trials()


    def convert_designs(self, dsns: np.ndarray):
        # convert q x d np.array designs back to their design object
        converted_dsns = [Design(dsn.tolist(), self.env.spec_range.keys()) for dsn in dsns]
        return converted_dsns

    def fn(self, *params):
        """Abstracts evaluation: takes in parameters for designs, and returns the cost of the first one??.
        """
        dsns = np.array([list(xpoint) for xpoint in params], dtype=np.int)
        dsn_objs = self.convert_designs(dsns)
        dsn_objs = self.env.evaluate(dsn_objs)
        self.db.extend(dsn_objs)
        res = dsn_objs[0]['cost']
        return res

    def start(self, max_eval=1000):
        best = fmin(self.fn, self.space, algo=tpe.suggest, max_evals=max_eval,
                    rstate=np.random.RandomState(self.seed),
                    trials=self.trials)
        best_params = space_eval(self.space, best)

    def save(self, fname: Path) -> None:
        db_pickl = self.db.picklable()
        state = dict(db=db_pickl, trials=self.trials)
        fname.parent.mkdir(parents=True, exist_ok=True)
        write_pickle(fname, state)

    def load(self, fname: Path) -> None:
        state = read_pickle(fname)
        self.db = state['db'].convert_to_database()
        self.trials = state['trials']
