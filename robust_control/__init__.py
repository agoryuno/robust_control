from itertools import product
import numpy as np

from numpy.linalg import svd
from torch.linalg import svd as tsvd
import torch

import cvxpy as cvx


DEFAULT_PART = False
DEFAULT_DENOISE = False


#@torch.no_grad()
@torch.jit.script
def bind_data_b(X: torch.Tensor):
    a, b = torch.min(X), torch.max(X)
    return (X - (a+b)/2.) / ((b-a)/2.), a, b


def bind_data(X):
    a, b = np.nanmin(X), np.nanmax(X)
    return (X - (a+b)/2.) / ((b-a)/2.), a, b


#@torch.no_grad()
@torch.jit.script
def unbind_data(X: torch.Tensor, a: float, b: float):
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


#@torch.no_grad()
@torch.jit.script
def compute_hat_p_b(Ys: torch.Tensor):
    # find the value of $\hat p$ in eq. (9) on page 8
    have_vals = torch.sum(~torch.isnan(Ys), (1,2))
    
    T = Ys.size(2)
    
    total_els = torch.numel(Ys[0, :, :])

    ps = torch.div(have_vals, float(total_els))
    ns = torch.full(ps.size(), 1/((Ys.size(1) - 1)*T))
    hat_ps = torch.maximum(ps, ns)
    return hat_ps


#@torch.no_grad()
@torch.jit.script
def get_ys_b(mat, treated_i: int):
    Y0 = torch.cat((mat[:treated_i, :], mat[treated_i+1:, :]), 0)
    Y1 = mat[treated_i, :]
    Y1 = Y1.view(1, Y1.size(0))
    return Y0, Y1


def get_ys(price_mat, treated_i):
    Y0 = np.vstack((price_mat[:treated_i, :], price_mat[treated_i+1:, :]))
    Y1 = price_mat[treated_i, :]
    Y1 = np.reshape(Y1, (1, Y1.shape[0]))
    return Y0, Y1


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


#@torch.no_grad()
@torch.jit.script
def partition(orig_tensor: torch.Tensor, parts: int):
    full_len = orig_tensor.size(orig_tensor.dim()-1)
    part_len = full_len // parts
    #idx = [(i*part_len, 
    #        j*part_len) 
    #            for i,j in zip(range(parts), range(1, 1+parts))]

    idx = [[i*(part_len - 1), part_len] for i in range(parts)]

    remain = full_len % parts
    if remain > 0:
        #idx[-1] = (idx[-1][0], full_len)
        idx[-1][1] = idx[-1][1]+remain

    new_tensor = torch.zeros((orig_tensor.size(0), orig_tensor.size(1), parts), 
        device=orig_tensor.device)
    for i, (start, end) in enumerate(idx):
        #new_tensor[:, i] = orig_tensor[:, start:end].mean(-1)
        t = torch.narrow(orig_tensor, -1, start, end).mean(-1)
        new_tensor[:, :, i] = t
    return new_tensor


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


#@torch.no_grad()
@torch.jit.script
def get_M_hat_b(Ys: torch.Tensor, mus: torch.Tensor, denoise: bool = DEFAULT_DENOISE):
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
    
    # Disable if denoise is False
    if denoise:
        s[s <= mus] = 0.

    # Make the singular values matrix
    smat = torch.zeros_like(Ys)
    b = torch.eye(s.size(1))
    c = s.unsqueeze(2).expand(s.size(0), s.size(1), s.size(1))
    smat[:, :c.size(2), :] = c * b

    # build the estimator of Y
    M_hat = u @ (smat @ v)
    m = torch.permute((1/hat_p).repeat(M_hat.size(2), 1, 1), (2,1,0))
    return m*M_hat


#@torch.no_grad()
@torch.jit.script
def estimate_weights_b(Y1: torch.Tensor, Y0: torch.Tensor, etas: torch.Tensor):
    assert etas.size(0) == Y0.size(0)
    res = tsvd(Y0, full_matrices=True)
    
    U, Vh, S = res.U, res.Vh, res.S
    
    D = torch.zeros_like(Y0)
    b = torch.eye(S.size(1), device=S.device)
    a = torch.div(S, torch.sub(torch.square(S), torch.square(etas)))
    c = a.unsqueeze(2).expand(a.size(0), a.size(1), a.size(1))
    D[:, :c.size(2), :] = c * b

    return U @ D @ Vh @ Y1.mT
    

#@torch.no_grad()
@torch.jit.script
def calc_control_b(Y1_t: torch.Tensor, Y0_t: torch.Tensor, 
                   etas: torch.Tensor, a: float, b: float):
    vs = estimate_weights_b(Y1_t, Y0_t, etas)
    
    Y1_hat = (Y0_t.mT@vs)
    Y1_hat = unbind_data(Y1_hat, a, b)
    Y0_t = unbind_data(Y0_t, a, b)
    return Y1_hat, Y0_t, vs


#@torch.no_grad()
torch.jit.script
def loss_fn(Y1s: torch.Tensor, Y1_hats: torch.Tensor):
    return torch.sum(torch.square(torch.sub(Y1s, Y1_hats)), 2)


def prepare_data(orig_mat, treated_i, etas, mus, denoise=DEFAULT_DENOISE):
    orig_tensor = torch.Tensor(orig_mat)
    
    batch_size = len(etas)*len(mus)
    etas_len = len(etas)

    mus = torch.Tensor(np.array(mus))
    mus = mus.reshape( (mus.size(0), 1))

    etas = torch.Tensor(np.array(etas))
    etas = etas.reshape( (etas.size(0), 1)).repeat(mus.size(0), 1)
    
    bound_mat, a, b = bind_data_b(orig_tensor)
    Y0, Y1 = get_ys_b(bound_mat, treated_i)

    y0 = Y0.repeat(mus.size(0),1,1)
    Y0_t = get_M_hat_b(y0, mus, denoise=denoise).repeat(etas_len, 1, 1)
    Y1_t = Y1.expand(batch_size, Y1.size(0), Y1.size(1))

    return Y1_t, Y0_t, etas, a, b


def get_control(orig_mat, treated_i, eta_n=10, mu_n=3, 
        cuda=False, parts=DEFAULT_PART, denoise=DEFAULT_DENOISE):
    """
    Given the matrix of values 'orig_mat' and the row index 
    'treated_i', computes synthetic controls for each combination
    of `eta` and `mu` for the respective numbers of `eta_n` and 
    `mu_n`.
    
    Returns a tensor of dimensions `orig_mat.size()[1] by 1` 
    that contains the synthetic control calculated with the best 
    found values of parameters $eta$ and $mu$, and a tensor with the
    same dimensions, containing the denoised original data for observation
    `treated_i`.
    """

    etas = np.logspace(-2, 3, eta_n).tolist()
    mus = [compute_mu(orig_mat, treated_i, w=w) 
        for w in np.linspace(0.1, 1., mu_n)]
    Y1_o, Y0_o, etas, a, b = prepare_data(orig_mat, treated_i, etas, mus, 
            denoise=denoise)
    Y1_t, Y0_t = Y1_o, Y0_o
    if parts:
        Y1_t = partition(Y1_t, parts)
        Y0_t = partition(Y0_t, parts)

    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Y1_t = Y1_t.to(device)
        Y0_t = Y0_t.to(device)
        etas = etas.to(device)
        a = a.to(device)
        b = b.to(device)

    vs = estimate_weights_b(Y1_t, Y0_t, etas)

    Y1_hats = unbind_data(vs.mT @ Y0_o, a, b)
    Y1s = unbind_data(Y1_o, a, b)

    assert (Y1_hats.size() == Y1s.size())
    
    min_idx = loss_fn(Y1s, Y1_hats).argmin()  
    res = Y1_hats[min_idx, :, :]

    # There's really no need to use the min_idx for the
    # denoised original data, but we have it so why not use it
    return res, Y1s[min_idx, :, :]


# TODO: Forward-chaining

if __name__ == "__main__":
    import pickle

    with open("price_mat.pkl", "rb") as f:
        price_mat, _, _ = pickle.load(f)

    treated_i = 0
    eta_n = 10
    mu_n = 3

    control, orig = get_control(price_mat, treated_i, eta_n, mu_n, cuda=False, parts=12, denoise=False)
    print (control/orig)