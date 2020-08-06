import argparse
import pdb

from utils.pdb import register_pdb_hook
from utils.importlib import import_class
from utils.file import read_yaml

register_pdb_hook()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Test black box environments')
    parser.add_argument('spec_file', type=str, help='the spec file for the black box environment')
    return parser.parse_args()


if __name__ == '__main__':
    _args = parse_arguments()

    specs = read_yaml(_args.spec_file)
    engine = import_class(specs['bb_engine'])(specs=specs['bb_engine_params'])

    # just generates the designs
    designs_empty = engine.generate_rand_designs(n=2, evaluate=False)
    # evaluates the design
    designs_populated = engine.evaluate(designs_empty, do_interpet=True)
    # generates the design and also evaluates it
    designs = engine.generate_rand_designs(n=2, evaluate=True)



    pdb.set_trace()
