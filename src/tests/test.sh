# Testing with multiple T values
python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 0.1 --max_nfe 10000 --run_name alpha=2.0_T=0.1

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 1.0 --max_nfe 10000 --run_name alpha=2.0_T=1.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 4.0 --max_nfe 10000 --run_name alpha=2.0_T=4.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 16.0 --max_nfe 10000 --run_name alpha=2.0_T=16.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 32.0 --max_nfe 10000 --run_name alpha=2.0_T=32.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 64.0 --max_nfe 10000 --run_name alpha=2.0_T=64.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 128.0 --max_nfe 10000 --run_name alpha=2.0_T=128.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 256.0 --max_nfe 10000 --run_name alpha=2.0_T=256.0

python3 run_GNN.py --block ext_laplacian --block attention --experiment --lr 0.0000001 --max_iters 1000 --time 512.0 --max_nfe 10000 --run_name alpha=2.0_T=512.0

