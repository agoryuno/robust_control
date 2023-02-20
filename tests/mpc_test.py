import pickle
import os
import sys
from multiprocessing import Pool

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_control import get_controls, split


with open("price_mat.pkl", "rb") as f:
    price_mat, _, _ = pickle.load(f)

print (os.cpu_count())        

eta_n = 10
mu_n = 3
i = 0
row_batches = split([i for i in range(price_mat.shape[0])], price_mat.shape[0]//10)

def fn(*args):
    price_mat, rows, eta_n, mu_n = args
    return get_controls(price_mat, rows, eta_n, mu_n=mu_n, cuda=False, 
                parts=False, preint=90, train=0.79, double=True)

args = [[price_mat, rows, eta_n, mu_n] for rows in row_batches]

def run():
    with Pool() as p:
        p.starmap(fn, args)

if __name__ == "__main__":
    run()