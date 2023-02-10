from itertools import product
import numpy as np

from numpy.linalg import svd
from torch.linalg import svd as tsvd
import torch

import cvxpy as cvx


def bind_data(X):
    a, b = np.nanmin(X), np.nanmax(X)
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

def compute_hat_p_b(Ys):
    # find the value of $\hat p$ in eq. (9) on page 8
    have_vals = torch.sum(~torch.isnan(Ys), (1,2))
    
    T = Ys.size()[2]
    
    total_els = torch.numel(Ys[0, :, :])
    ps = torch.div(have_vals.to(torch.float32), float(total_els))
    ns = torch.Tensor([1/((Ys.size(1)-1)*T)]*ps.size(0))
    hat_ps = torch.maximum(ps, ns)
    return hat_ps


def get_M_hat_b(Ys, mus):
    """
    Returns the estimator of Y: M_hat
    """
    hat_p = compute_hat_p_b(Ys)
    
    # fill in missing values
    Ys = torch.nan_to_num(Ys)

    # compute the SVD of $Y$
    res = tsvd(Ys, full_matrices=True)
    
    u,s,v = res.U, res.S, res.Vh

    # Remove singular values that are below $\mu$
    # by setting them to zero
    mus = torch.Tensor(np.array(mus).reshape( (s.size(0),1)))
    s[s <= mus] = 0.

    # Make the singular values matrix
    smat = torch.zeros(Ys.size())
    b = torch.eye(s.size(1))
    c = s.unsqueeze(2).expand(*s.size(), s.size(1))
    smat[:, :c.size(2), :] = c * b
    smat = smat.to(torch.float64)

    # build the estimator of Y
    M_hat = u @ (smat @ v)
    m = torch.permute((1/hat_p).repeat(M_hat.size(2), 1, 1), (2,1,0))
    return m*M_hat


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
    res = svd(Y0, full_matrices=True)

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


def estimate_weights_b(Y1, Y0, etas):
    assert len(etas) == Y0.size()[0]
    res = tsvd(Y0, full_matrices=True)
    
    U, Vh, S = res.U, res.Vh, res.S
    
    D = torch.zeros(Y0.size(), dtype=torch.float64)
    a = torch.div(S, torch.sub(torch.square(S), torch.square(etas)))
    
    for i in range(a.size(0)):
        D[i, :Y0.size()[2], :] = torch.diag(a[i])
    return U @ D @ Vh @ Y1
    

def calc_control_b(orig_mat, treated_i, etas, mus, device=None):
    batch_size = len(etas)*len(mus)
    
    bound_mat, a, b = bind_data(orig_mat)
    
    Y0, Y1 = get_ys(bound_mat, treated_i)

    y0 = torch.from_numpy(Y0).repeat(len(mus),1,1).to(torch.float64)
    if device:
        y0 = y0.to(device)

    Y0_t = get_M_hat_b(y0, mus).repeat(len(etas), 1, 1)
    Y1_t = torch.from_numpy(Y1.T).repeat(batch_size, 1, 1).to(torch.float64)
    if device:
        Y1_t = Y1_t.to(device)

    assert isinstance(etas, list)
    etas = etas*len(mus)
    etas = np.array(etas)
    etas = np.reshape(etas, (etas.shape[0], 1))
    etas = torch.Tensor(etas).to(torch.float64)
    if device:
        etas = etas.to(device)
    vs = estimate_weights_b(Y1_t, Y0_t, etas)
    
    Y1_hat = (Y0_t.mT@vs)
    Y1_hat = unbind_data(Y1_hat, a, b)
    Y0_t = unbind_data(Y0_t, a, b)
    return Y1_hat, Y0_t, vs


def loss_fn(Y1s, Y1_hats):
    return torch.sum(torch.square(torch.sub(Y1s, Y1_hats)), 1)


def make_Y1s(orig_mat, i, n):
    """
    Makes an `n by k by j` tensor by repeating
    `orig_mat[i]` n times, with `orig_mat[i]`
    having dimensions `k by j`
    """
    mat_t = torch.Tensor(orig_mat[i])
    mat_t = torch.reshape(mat_t, (mat_t.size()[0], 1))
    return mat_t.repeat(n, 1, 1)


def get_control(orig_mat, treated_i, eta_n, mu_n, device=None):
    """
    Given the matrix of values 'orig_mat' and the row index 
    'treated_i', computes synthetic controls for each combination
    of `eta` and `mu` for the respective numbers of `eta_n` and 
    `mu_n`.
    
    Returns a tensor of dimensions `orig_mat.size()[1] by 1` that contains
    the synthetic control calculated with the best found values of 
    parameters $eta$ and $mu$.
    """
    etas = np.logspace(-2, 3, eta_n).tolist()
    mus = [compute_mu(orig_mat, treated_i, w=w) for w in np.linspace(0.1, 1., mu_n)]
    Y1_hats, Y0s, vs = calc_control_b(orig_mat, treated_i, etas, mus, device=device)
    Y1s = make_Y1s(orig_mat, treated_i, Y1_hats.size()[0])
    return Y1_hats[loss_fn(Y1s, Y1_hats).argmin(), :, :]