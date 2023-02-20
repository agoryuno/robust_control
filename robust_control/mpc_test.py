import pickle
import os
import sys
from time import time
from multiprocessing import Pool

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_control import get_controls, split


def make_arguments(batch_size=20, eta_n=10, mu_n=3, preint=30,
                   time_len=45, cuda=False, double=False):
    
    assert preint < time_len
    with open("price_mat.pkl", "rb") as f:
        price_mat, _, _ = pickle.load(f)

    row_batches = split([i for i in range(price_mat[:, :time_len].shape[0])], price_mat.shape[0]//batch_size)
    return [[price_mat, rows, eta_n, mu_n, preint, cuda, double] for rows in row_batches]


def fn(*args):
    price_mat, rows, eta_n, mu_n, preint, cuda, double = args
    return get_controls(price_mat, rows, eta_n, mu_n=mu_n, cuda=cuda, 
                parts=False, preint=preint, train=0.8, double=double)


def run(args, procs=None):
    with Pool(procs) as p:
        p.starmap(fn, args)


if __name__ == "__main__":
    
    args = make_arguments(30)
    start = time()
    run(args, procs=4)
    print (time() - start)