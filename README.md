# robust_control

A PyTorch implementation of the "robust" synthetic control model proposed by Amjad, Shah and Shen \[[arxiv](https://arxiv.org/abs/1711.06940)\].

The robust synthetic control is an extension of the original synthetic control model, proposed by Abadie, Diamond and Hainmueller (for overview see:
https://www.aeaweb.org/articles?id=10.1257/jel.20191450).

The main advantages of the robust model are its ability to handle missing data and its speed. The original model involves a
non-convex optimization which can make parameter estimation take too long, especially when many synthetic controls need to be generated. The robust model,
on the other hand, utilizes a ridge regression with a scalar regularization parameter, which makes it possible to use singular value decomposition for
estimation.

The package currently implements most of the optimizations suggested by Amjad et al for their Algorithm 1 (the non-Bayesian variant), with the notable exception of forward-chaining for
hyperparameter selection - the standard train-test split is used instead.

The package is in a very early pre-alpha for now. That is to mean that it works correctly until a commit stops it from doing so until it works
again :)

# Installation

No distributions are provided yet, so installing directly from the repo is the only option:

`pip install git+https://github.com/agoryuno/robust_control`

# Usage

The package provides two main functions: `get_control` and `get_controls`. The former is intended to be used for estimating a single row of a matrix, while `get_controls()` is for multiple rows.
They are kept separate because PyTorch's heuristics result in subtle differences in the results,
 depending on batch size. In general, if you only want to get a control estimate for a single object, use `get_control()`, otherwise use `get_controls()` as it offers significant performance
 improvement.

## Function `get_control()`

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
orig_mat: np.ndarray
    The data matrix

treated_i: int
    The index of the treated object (row)

eta_n: int
    (Optional) The number of values of $eta$ to use
mu_n: Union[int, Literal[False]]
    (Optional) The number of values of $mu$ to use (anything over 5 is useless, default is False
    which means a single value))

cuda: bool
    (Optional) Whether to use CUDA - CUDA support for SVD in PyTorch is limited so this is
    best left at default value of False

parts: int
    (Optional) The number of partitions to use for training or False to not use partitions
    (default is False)

preint: bool
    (Optional) Number of pre-intervention periods to estimate the control on, if False
    uses all periods (default is False)

train: float
    (Optional) The proportion of the data to use for training (default is 0.8)

Returns:
-----------

Y1_o: torch.Tensor
    A tensor of shape (orig_mat.shape[1], 1) containing the original data for 
    the treated object

Y0_o: torch.Tensor
    A tensor of shape (orig_mat.shape[1], 1) containing the synthetic control
    data

v: torch.Tensor
    A tensor of shape (orig_mat.shape[0]-1, 1) containing the weights of the synthetic control


### Note on CUDA and performance:

Unfortunately, PyTorch's implementation of SVD isn't fully parallelizable on CUDA, so running the
estimation on the GPU may actually be slower than on the CPU (see this issue for details: https://github.com/pytorch/pytorch/issues/41306). For that reason CUDA support is disabled by default
but remains an option for the future.

As an alternative, there is in example of using `multiprocessing` for parallel estimation of
all rows in a matrix in `tests/mpc_test.py`.