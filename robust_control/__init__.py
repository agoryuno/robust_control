from itertools import product
import numpy as np
from typing import Literal, Optional, Union, List, Tuple

from numpy.linalg import svd
from torch.linalg import svd as tsvd
import torch

import cvxpy as cvx


DEFAULT_PART = False
DEFAULT_DENOISE = 3


#@torch.no_grad()
@torch.jit.script
def bind_data_b(X: torch.Tensor):
    X_min = torch.nan_to_num(X, nan=np.inf)
    a = torch.min(X_min)
    del X_min
    X_max = torch.nan_to_num(X, nan=-np.inf)
    b = torch.max(X_max)
    return (X - (a+b)/2.) / ((b-a)/2.), a, b


def bind_data(X: torch.Tensor):
    a, b = X.min(), X.max()
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


def compute_hat_p(Y: torch.Tensor):
    # find the value of $\hat p$ in eq. (9) on page 8
    have_vals = Y[~np.isnan(Y)].dim()
    
    T = Y.shape[1]
    
    p = have_vals/Y.dim()
    return np.maximum(p, 1/((Y.shape[0]-1)*T))

@torch.jit.script
def compute_hat_p_bb(Ys: torch.Tensor):
    # find the value of $\hat p$ in eq. (9) on page 8
    have_vals = torch.sum(~torch.isnan(Ys), (2,3))
    
    T = Ys.shape[-1]
    
    total_els = torch.numel(Ys[0, 0, :, :])
    ps = torch.div(have_vals, total_els)

    ns = torch.full(ps.size(), 1/((Ys.size(1) - 1)*T))
    hat_ps = torch.maximum(ps, ns)
    return hat_ps

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


@torch.jit.script
def get_ys_bb(mat: torch.Tensor, 
              rows_idx: torch.Tensor,
              other_idx: torch.Tensor):
    # Split the tensor `mat` into two tensors, one containing the rows
    # specified in `rows` and the other containing the remaining rows.
    # The rows are placed in a new dimension at the front of the tensors.
    Y1 = torch.index_select(mat, 0, rows_idx).view(rows_idx.shape[0], 1, 1, mat.shape[1])
    Y0 = mat.unsqueeze(0).expand(rows_idx.shape[0], mat.shape[0], mat.shape[1])
    Y0 = torch.masked_select(Y0, other_idx).view(rows_idx.shape[0], 
                                                 mat.shape[0]-1, 
                                                 mat.shape[1])
    Y0 = Y0.unsqueeze(1)
    assert Y0.shape[0] == Y1.shape[0]
    assert Y0.dim() == 4 == Y1.dim()
    return Y0, Y1


def get_ys(price_mat: torch.Tensor, treated_i:int):
    if isinstance(price_mat, np.ndarray):
        price_mat = torch.from_numpy(price_mat)
    Y0 = torch.cat((price_mat[..., :treated_i, :], price_mat[..., treated_i+1:, :]), -2)
    Y1 = price_mat[..., treated_i, :]
    Y1 = Y1.unsqueeze(-2)
    return Y0, Y1


def compute_sigma(price_mat: torch.Tensor, treated_i, preint_len=None):
    if isinstance(price_mat, np.ndarray):
        price_mat = torch.from_numpy(price_mat)
    preint_len = preint_len if preint_len else price_mat.shape[1]
    Y_t = price_mat[..., treated_i, :preint_len]
    hat_Y = torch.mean(price_mat[..., :, :preint_len], axis=0)
    a = (1./(preint_len-1)) 
    b = torch.sum((Y_t - hat_Y)**2)
    
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


@torch.jit.script
def get_M_hat_bb(
                Ys: torch.Tensor,
                mus: torch.Tensor,
                denoise: bool = DEFAULT_DENOISE
                ):
    """
    Returns the estimator of Y: M_hat
    """
    hat_p = compute_hat_p_bb(Ys)

    # fill in missing values in Ys
    Ys = torch.nan_to_num(Ys)

    # compute the SVD of `Ys`
    res = tsvd(Ys, full_matrices=True)

    u,s,v = res.U, res.S, res.Vh

    # Remove singular values that are below $\mu$
    # by setting them to zero
    
    s[s <= mus.unsqueeze(-1)] = 0.


    # Make the singular values matrix
    smat = torch.zeros_like(Ys)
    b = torch.eye(s.shape[-1], device=s.device)
    c = s.unsqueeze(s.dim()).expand(s.shape[0], s.shape[1], s.shape[2], s.shape[2])
    smat[:, :, :c.shape[-1], :] = c * b

    # build the estimator of Y
    M_hat = u @ (smat @ v)
    m = torch.permute((1/hat_p).repeat(M_hat.shape[-1], 1, 1, 1), (2, 3, 1, 0))
    return m*M_hat


#@torch.no_grad()
#@torch.jit.script
def get_M_hat_b(
                Ys: torch.Tensor,
                mus: torch.Tensor
                ):
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

    s[s <= mus] = 0.

    # Make the singular values matrix
    smat = torch.zeros_like(Ys)
    b = torch.eye(s.size(1))
    c = s.unsqueeze(s.dim()).expand(s.shape[-2], s.shape[-1], s.shape[-1])
    smat[:, :c.shape[-1], :] = c * b

    # build the estimator of Y
    M_hat = u @ (smat @ v)
    m = torch.permute((1/hat_p).repeat(M_hat.size(2), 1, 1), (2,1,0))
    return m*M_hat


#@torch.no_grad()
@torch.jit.script
def estimate_weights_b(Y1: torch.Tensor, Y0: torch.Tensor, etas: torch.Tensor):
    assert etas.size(0) == Y0.size(0), (etas.size(0), Y0.size(0))

    res = tsvd(Y0, full_matrices=True)
    
    U, Vh, S = res.U, res.Vh, res.S
    
    D = torch.zeros_like(Y0)
    b = torch.eye(S.size(1), device=S.device)
    a = torch.div(S, torch.sub(torch.square(S), torch.square(etas)))
    c = a.unsqueeze(2).expand(a.size(0), a.size(1), a.size(1))
    D[:, :c.size(2), :] = c * b

    return U @ D @ Vh @ Y1.mT
    

@torch.jit.script
def estimate_weights_bb(Y1: torch.Tensor, 
                        Y0: torch.Tensor, 
                        etas: torch.Tensor
                        ):
    assert etas.size(0) == Y0.size(0), (etas.size(0), Y0.size(0))

    res = tsvd(Y0, full_matrices=True)

    U, Vh, S = (res.U, 
                res.Vh, 
                res.S)
    
    D = torch.zeros_like(Y0, device=S.device)
    
    b = torch.eye(S.shape[-1], device=S.device)
    a = torch.div(S, torch.sub(torch.square(S), torch.square(etas)))
    c = a.unsqueeze(S.dim()).expand(a.size(0), a.size(1), a.size(2), a.size(2))
    D[:, :, :c.shape[-1], :] = c * b
    result = U @ D @ Vh @ Y1.mT
    return result


#@torch.no_grad()
torch.jit.script
def loss_fn(Y1s: torch.Tensor, Y1_hats: torch.Tensor):
    return torch.sum(torch.square(torch.sub(Y1s, Y1_hats)), -1)


def prepare_data_b( 
                    orig_mat: torch.Tensor,
                    rows: List,
                    etas: torch.Tensor,
                    mu_n: int,
                    double: bool = False
                    ):
    
    if double:
        orig_mat = orig_mat.double()
        
    bound_mat, a, b = bind_data_b(orig_mat)

    mus = np.array([0.5]*len(rows)).reshape((len(rows), 1))
    if mu_n:
        # Create a torch.Tensor of `mu_n` $mu$ values for each row in `rows`
        # with shape (len(rows), mu_n)
        mus = np.array([[compute_mu(bound_mat, i, w=w) 
                         for w in np.linspace(0.1, 1., mu_n)] for i in rows])
    mus = torch.from_numpy(mus)

    etas = etas.repeat(1, mus.shape[-1]).unsqueeze(0)

    if double:
        etas = etas.double()
        mus = mus.double()

    rows_idx = torch.tensor(rows, device=orig_mat.device)
    other_idx = torch.full_like(orig_mat, True, dtype=torch.bool)
    other_idx = other_idx.unsqueeze(0).repeat(rows_idx.shape[0], 1, 1)
    for i,row in enumerate(rows_idx):
        other_idx[i, row.item(), :] = False
    Y0, Y1 = get_ys_bb(bound_mat, rows_idx, other_idx)

    y0 = Y0.repeat(1, mus.size(1), 1, 1) 

    

    M_hat = get_M_hat_bb(y0, mus)
    
    Y0_t = M_hat.repeat(1, etas.shape[-1]//mus.shape[-1], 1, 1)

    Y1_t = Y1.expand(Y1.shape[0], etas.shape[-1], Y1.size(2), Y1.size(3))

    etas = etas.permute(1, 2, 0)
    return Y1_t, Y0_t, etas, a, b


def prepare_data(orig_mat: torch.Tensor, 
                 treated_i: int, 
                 etas: List, 
                 mu_n: int):
    orig_tensor = torch.Tensor(orig_mat)
    
    batch_size = len(etas)*mu_n
    etas_len = len(etas)

    etas = torch.Tensor(np.array(etas))
    etas = etas.reshape( (etas.size(0), 1)).repeat(mu_n, 1)
    
    bound_mat, a, b = bind_data_b(orig_tensor)
    Y0, Y1 = get_ys_b(bound_mat, treated_i)

    y0 = Y0.repeat(mu_n,1,1)

    mus = [0.5]
    if mu_n:
        mus = [compute_mu(bound_mat, treated_i, w=w) 
            for w in np.linspace(0.1, 1., mu_n)]

    mus = torch.from_numpy(np.array(mus)).unsqueeze(1)

    M_hat = get_M_hat_b(y0, mus)
    
    Y0_t = M_hat.repeat(etas_len, 1, 1)

    Y1_t = Y1.expand(batch_size, Y1.size(0), Y1.size(1))
    return Y1_t, Y0_t, etas, a, b


@torch.jit.script
def _get_train_data(Y1_o, Y0_o, cutoff: int, parts: int):
    Y1_t, Y0_t = Y1_o[..., :cutoff], Y0_o[..., :cutoff]
    if parts > 0:
        Y1_t = partition(Y1_t, parts)
        Y0_t = partition(Y0_t, parts)
    return Y1_t, Y0_t


def _make_params(
        mat: np.ndarray,  
        eta_n: int, 
        preint: Union[Literal[False], int], 
        treated_i: Optional[int] = None, 
        parts: Optional[int] = None) -> Tuple[List, torch.Tensor, int, int, bool]:
    """
    Prepares hyperparameters for the `get_control()` and `get_all_controls()`
    functions.

    Parameters:
    -----------
    mat: np.ndarray
        The data matrix
    treated_i: int
        The index of the treated object (row)
    eta_n: int
        The number of values of eta to use
    mu_n: int
        The number of values of mu to use
    preint: int
        The number of observations (columns) to use for training
    parts: int
        The number of partitions to use for training 
        (default is None)
    
    Returns:
    -----------
    etas: torch.Tensor
        A tensor of shape (eta_n, 1) containing the values of eta
    mus: torch.Tensor
        A tensor of shape (mu_n, 1) containing the values of mu
    cutoff: int
        The number of observations to use for training
    parts: int
        The number of partitions to use for training (0 if no partitions)
    denoise: bool
        Whether to denoise the data
    """

    etas = np.logspace(-2, 3, eta_n).tolist()
    

    cutoff = mat.shape[1]
    if preint:
        cutoff = preint

    parts = 0 if not parts else parts
    return etas, cutoff, parts


def _make_params_b(mat: np.ndarray,  
        eta_n: int, 
        preint: Union[Literal[False], int], 
        rows: Optional[List] = None, 
        parts: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int, int, bool]:
    
    # Generate `eta_n` $eta$ values for each row in `rows`
    # convert to a torch.Tensor and 
    # reshape it to (len(rows), eta_n)
    etas = np.logspace(-2, 3, eta_n)
    etas = torch.from_numpy(etas).unsqueeze(0).expand(len(rows), eta_n).type(torch.float32)

    cutoff = mat.shape[1]
    if preint:
        cutoff = preint
    
    parts = 0 if not parts else parts
    
    return etas, cutoff, parts


def get_control(orig_mat: torch.Tensor, 
                treated_i: int, 
                eta_n: Optional[int] = 10, 
                mu_n: Optional[Union[int, Literal[False]]] = DEFAULT_DENOISE, 
                cuda: Optional[bool] = False, 
                parts: Optional[Union[int, Literal[False]]] = DEFAULT_PART, 
                preint: Optional[bool] = False, 
                train: Optional[float] = .8):
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

    Parameters:
    -----------
    `orig_mat`: np.ndarray
        The data matrix

    `treated_i`: int
        The index of the treated object (row)
    
    `eta_n`: int (optional)
        The number of values of $eta$ to use

    `mu_n`: Union[int, Literal[False]] (optional)
        The number of values of $mu$ to use (anything over 5 is useless, default is False
        which means a single value))
    
    `cuda`: bool (optional)
        Whether to use CUDA - CUDA support for SVD in PyTorch is limited so this is
        best left at default value of False
    
    `parts`: int (optional)
        The number of partitions to use for training or False to not use partitions
        (default is False)
    
    `preint`: bool (optional)
        Number of pre-intervention periods to estimate the control on, if False - uses 
        all periods (default is False)
    
    `train`: float (optional)
        The proportion of the data to use for training (default is 0.8)
    
    Returns:
    -----------
    
    `Y1_c`: torch.Tensor
        A tensor of shape (orig_mat.shape[1], 1) containing the synthetic control data for 
        the treated object
    
    `Y0_o`: torch.Tensor
        A tensor of shape (orig_mat.shape[1], 1) containing the original
        data
    
    `v`: torch.Tensor
        A tensor of shape (orig_mat.shape[0]-1, 1) containing the weights of the synthetic control.
        Keep in mind that these weights are calculated for "normalized" data, so applying them
        to untransformed original data will yield incorrect results. See next for an example
        of using the weights to calculate the synthetic control.


    Using the weights:
    ------------------

    This is an example of using the `v` matrix of weights returned by the `get_control()`
    function. Assuming that your original data is in `orig_mat` and the index of the treated
    object is `treated_i`:

    ```python
    from robust_control import get_control, bind_data_b, unbind_data

    # Calculate the synthetic control
    Y1_c, Y0_o, v = get_control(orig_mat, treated_i)

    # Bind the data for untreated objects
    dat, a, b = bind_data_b(torch.concat ((orig_mat[:treated_i], orig_mat[treated_i+1:])))

    # Apply the weights to `dat`
    control = dat.T @ v

    # Unbind the data to bring it back to the original scale
    control = unbind_data(control, a, b)
    
    ```
    """

    etas, cutoff, parts = _make_params(orig_mat,
                                       eta_n,
                                       preint=preint,
                                       parts=parts,
                                       treated_i=treated_i)
    
    Y1_o, Y0_o, etas, a, b = prepare_data(orig_mat, treated_i, etas, mu_n)


    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Y1_o = Y1_o.to(device)
        Y0_o = Y0_o.to(device)
        etas = etas.to(device)
        a = a.to(device)
        b = b.to(device)

    train_i = int(np.floor(train*cutoff))
    Y1_t, Y0_t = _get_train_data(Y1_o, Y0_o, train_i, parts)

    vs = estimate_weights_b(Y1_t, Y0_t, etas)

    Y1_eta = vs.mT @ Y0_o
    
    min_idx = loss_fn(Y1_o[:, :, train_i:cutoff], Y1_eta[:, :, train_i:cutoff]).argmin()
    
    Y1_n, Y0_n = _get_train_data(Y1_o[min_idx, :, :].unsqueeze(0), 
                                 Y0_o[min_idx, :, :].unsqueeze(0), 
                                 cutoff, parts)
    vs = estimate_weights_b(Y1_n, Y0_n, etas[min_idx].unsqueeze(0))

    Y1_hats = unbind_data(vs.mT @ Y0_o, a, b)
    Y1s = unbind_data(Y1_o, a, b)

    assert (Y1_hats.size() == Y1s.size())

    return Y1_hats[0], Y1s[0], vs[0]


# A function to get controls for a set of rows
def get_controls(orig_mat, rows: Optional[List] = None, eta_n=10, mu_n=DEFAULT_DENOISE, 
        cuda=False, parts=DEFAULT_PART, preint=False, train: float = 1.,
        double: bool = False):
    """
    Given the matrix of values 'orig_mat', computes synthetic 
    controls for all rows in `rows`. If `rows` is None, computes
    synthetic controls for all rows in `orig_mat`.
    
    Returns a tensor with dimensions `orig_mat.shape[0] by orig_mat.shape[0] by
     orig_mat.shape[1]` that contains the synthetic controls calculated with the best 
    found values of parameters $eta$ and $mu$ for each row, and a tensor with
    the same dimensions as `orig_mat` containing the denoised original data.
    """
    raise NotImplementedError("This function is currently disabled")

    # Compared to `get_control()`, this function adds a new batch dimension
    # that holds batches of rows from the original matrix. Each batch is
    # is a 3D tensor like the one used by `get_control()`.
    etas, cutoff, parts = _make_params_b(orig_mat, 
                                         eta_n, 
                                         preint=preint, 
                                         rows=rows,
                                         parts=parts
                                         )
    Y1_o, Y0_o, etas, a, b = prepare_data_b(torch.Tensor(orig_mat), 
                                            rows, 
                                            etas, 
                                            mu_n, 
                                            double=double)
    
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Y1_o = Y1_o.to(device)
        Y0_o = Y0_o.to(device)
        etas = etas.to(device)
        a = a.to(device)
        b = b.to(device)

    train_i = int(np.floor(train*cutoff))
    Y1_t, Y0_t = _get_train_data(Y1_o, Y0_o, train_i, parts)

    vs = estimate_weights_bb(Y1_t, Y0_t, etas)

    Y1_eta = vs.mT @ Y0_o

    loss = loss_fn(Y1_o[:, :, :, train_i:cutoff], Y1_eta[:, :, :, train_i:cutoff])
    min_idx = loss.argmin(dim=1)

    Y1_n, Y0_n = _get_train_data(Y1_o[:, min_idx[:,0]], 
                                 Y0_o[:, min_idx[:,0]], 
                                 cutoff, parts)
    
    vs = estimate_weights_bb(Y1_n, Y0_n, etas[:, min_idx[:,0]])

    Y1_hats = unbind_data(vs.mT @ Y0_o[:, min_idx[:,0]], a, b)
    Y1s = unbind_data(Y1_o[:, min_idx[:,0]], a, b)

    assert (Y1_hats.size() == Y1s.size())

    return Y1_hats[0], Y1s[0], vs[0]

# TODO: Forward-chaining

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    import pickle

    with open("price_nan.pkl", "rb") as f:
        price_nan = pickle.load(f)

    with open("price_mat.pkl", "rb") as f:
        price_mat, _, _ = pickle.load(f)


    treated_i = 1
    eta_n = 10
    mu_n = 3
    #rows = [0,1]
    """row_batches = split([i for i in range(price_mat.shape[0])], price_mat.shape[0]//10)
    from tqdm import tqdm 

    for rows in tqdm(row_batches):
        control, orig, v = get_controls(price_mat[:, :45], 
                                    rows, 
                                    eta_n, 
                                    preint=35,
                                    mu_n=mu_n, 
                                    cuda=False,)
    print (control/orig)
    #b = control[0]/orig[0]"""
