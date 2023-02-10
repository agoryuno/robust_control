from itertools import product
import numpy as np

from numpy.linalg import svd
from torch.linalg import svd as tsvd
import torch

import cvxpy as cvx


def bind_data(X):
    a, b = X.min(), X.max()
    return (X - (a+b)/2.) / ((b-a)/2.), a, b


def unbind_data(X, a, b):
    return X * (b-a)/2. + (a + b)/2.


def get_mu_schedule(price_mat, treated_i):
    """
    Returns a list of mu values in ascending order,
    each removing one additional singular value.
    """
    bound_mat, a, b = bind_data(price_mat)
    Y = np.vstack((bound_mat[:treated_i, :], bound_mat[treated_i+1:, :]))
    _,s,_ = svd(Y, full_matrices=True)
    return sorted(s)


def compute_hat_p(Y):
    # find the value of $\hat p$ in eq. (9) on page 8
    have_vals = Y[~np.isnan(Y)].size
    
    T = Y.shape[1]
    
    p = have_vals/Y.size
    return np.maximum(p, 1/((Y.shape[0]-1)*T))


def get_M_hat(Y, mu=0.):
    """
    Returns the estimator of Y: M_hat
    """
    hat_p = compute_hat_p(Y)
    # fill in the missing values as per eq. (7) on page 8
    Y[np.isnan(Y)] = 0
    
    # compute the SVD of $Y$
    u,s,v = svd(Y, full_matrices=True)

    # Remove singular values that are below $\mu$
    s = s[s >= mu]
    
    smat = np.zeros((Y.shape[0], Y.shape[1]))
    smat[:s.shape[0], :s.shape[0]] = np.diag(s)
    
    # build the estimator of Y
    M_hat = np.dot(u, np.dot(smat, v))
    return (1/hat_p)*M_hat


def estimate_weights(Y1, Y0, eta=0.6):
    res = tsvd(Y0, full_matrices=True)

    D = torch.zeros(Y0.size())
    D[:Y0.size()[1], :] = torch.diag(torch.div(res.S, torch.sub(torch.square(res.S), eta**2)))
   
    v = res.U @ D @ res.Vh @ Y1.T
    return v.numpy()


def get_ys(price_mat, treated_i):
    Y0 = np.vstack((price_mat[:treated_i, :], price_mat[treated_i+1:, :]))
    Y1 = price_mat[treated_i, :]
    Y1 = np.reshape(Y1, (1, Y1.shape[0]))
    return Y0, Y1
    

def denoise(price_mat, mu):
    bound_mat, a, b = bind_data(price_mat)
    M_hat = get_M_hat(bound_mat, mu)
    return unbind_data(M_hat, a, b)


def calc_control(price_mat, treated_i, preint_len=None, eta=0.0, mu=0.4):
    preint_len = preint_len if preint_len else price_mat.shape[1]
    bound_mat, a, b = bind_data(price_mat)
    
    Y0, Y1 = get_ys(bound_mat, treated_i)
    M_hat = get_M_hat(Y0, mu)
    
    v = estimate_weights(torch.Tensor(Y1[:, :preint_len]), torch.Tensor(M_hat[:, :preint_len]), eta)
    
    Y1_hat = (M_hat.T@v).T

    hat_data = np.vstack((M_hat[:treated_i, :], Y1_hat, M_hat[treated_i:, :]))
    hat_data = unbind_data(hat_data, a, b)
    return hat_data, v


def compute_sigma(price_mat, treated_i, preint_len=None):
    preint_len = preint_len if preint_len else price_mat.shape[1]
    Y_t = price_mat[treated_i, :preint_len]
    hat_Y = np.mean(price_mat[:, :preint_len], axis=0)
    a = (1./(preint_len-1)) 
    b = np.sum((Y_t - hat_Y)**2)
    
    return a*b


def compute_mu(price_mat, treated_i, preint_len=None, w=None):
    preint_len = preint_len if preint_len else price_mat.shape[1]
    price_mat, _,_ = bind_data(price_mat)
    sigma = compute_sigma(price_mat, treated_i, preint_len)
    Y0, _ = get_ys(price_mat, treated_i)
    p = compute_hat_p(Y0)
    if not w:
        w = np.random.uniform(0.1, 1)
    return (2 + w) * np.sqrt(price_mat.shape[1] * (sigma*p + p*(1-p)))


def optimize_eta(price_mat, treated_i, preint=None, eta_num=10, 
                  base_w=0.5, mu_tries=None,
                  start=2):
    assert start >= 2
    etas= np.logspace(-2, 3, eta_num)
    
    def loss_fn(X, Y):
        return cvx.pnorm(X - Y, p=2)**2
    
    min_loss = np.inf
    
    mus = [compute_mu(price_mat, treated_i, preint, base_w)]
    if mu_tries:
        mus = [compute_mu(price_mat, treated_i, preint, w) for w in np.linspace(0.1, 1., mu_tries)]
    
    best_eta = None
    best_mu = None
    for eta,mu in product(etas,mus):
        loss = 0.
        for i in range(start, price_mat.shape[1]):
            try:
                hat_data, _ = calc_control(price_mat, treated_i, 
                                       preint_len=i-1, eta=eta, mu=mu)
            except ValueError:
                continue
            Y_hat = hat_data[treated_i, i]
            Y = price_mat[treated_i, i]
            loss += loss_fn(Y_hat, Y).value
        if loss < min_loss:
            min_loss = loss
            best_eta = eta
            best_mu = mu
            
    return best_eta, best_mu


def partition(price_mat, parts):
    preint = price_mat.shape[1]
    idx = [(i*(preint//parts), j*(preint // parts)) for i,j in zip(range(parts), range(1, 1+parts))]

    remain = preint % parts
    if remain > 0:
        idx[-1] = (idx[-1][0], preint)
    
    return np.vstack([np.mean(price_mat[:, i:j], axis=1) for i,j in idx]).T


def filter_by_transactions(data, top_n=300):
    a = data[["ticker", "transactions"]].groupby("ticker").sum().reset_index()
    tickers = a.sort_values("transactions").iloc[-top_n:]["ticker"]
    return data[data["ticker"].isin(tickers)]


def clean_anomalies(vec, threshold=10):
    vec = vec.copy()
    stds = np.array([vec[:i].std() for i in range(1,vec.shape[0])])
    c = np.abs(stds[1:]/stds[:-1])
    i = np.where((c >= threshold) & (c != np.inf)) [0] + 1
    
    mask = np.full(vec.shape[0], True)
    mask[i] = False
    mult = vec[i]/vec[mask].mean()
    
    diff = vec[i] - vec[mask].mean()
    sign = diff/np.abs(diff)
    vec[i] = sign*(vec[mask].mean()+mult*vec[mask].std())
    return vec

def calc_rmspe(fact, control, preint):
    pre = np.mean( (fact[:preint] - control[:preint])**2)
    post = np.mean( (fact[preint:] - control[preint:])**2)
    return post/pre
