import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--function", required=False, default="ext_laplacian3", help="The extended laplacian function to use")
parser.add_argument("--clip_low", required=False, type = float, default=0.05, help="Lower bound for clipping value")
parser.add_argument("--clip_high", required=False, type = float,  default=0.95, help="Upper bound for clipping value")
parser.add_argument("--clip_step", required=False, default=0.1, help="Step size for clipping values")
args = vars(parser.parse_args())

alphas = [1.0, 2.0, 3.0, 4.0]
# alphas = [1.0]
bounds = np.arange(args['clip_low'], args['clip_high'], args['clip_step'])

cmd = """
    python3 run_GNN.py --function {}
                       --block attention 
                       --experiment 
                       --lr 0.0000001 
                       --max_iters 1000 
                       --time 128.0 
                       --max_nfe 100000000 
                       --run_name 'Clipping alpha={} - bound={}'
                       --alpha {} 
                       --clip_bound {}
"""

for alpha in alphas:
    for bound in bounds:
        cmd_ = cmd.format(args['function'], alpha, bound, alpha, bound).replace("\n", "").replace("\t", "")
        # os.system(cmd_)
        print(cmd_)
