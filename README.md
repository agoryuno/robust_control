# robust_control

A PyTorch implementation of the "robust" synthetic control model proposed by Amjad, Shah and Shen \[[arxiv](https://arxiv.org/abs/1711.06940)\].

The robust synthetic control is an extension of the original synthetic control model, proposed by Abadie, Diamond and Hainmueller (for overview see:
https://www.aeaweb.org/articles?id=10.1257/jel.20191450). The synthetic control method is a causal inference method that uses a panel of data to estimate the effect of an intervention on one member
of that panel. The synthetic control for a "treated" object is constructed as a weighted average of
all other objects in the panel, with weights estimated controling for relevant covariates. The robust model is an extension of the original model that doesn't use covariates and instead fits
the weights based on the data itself. It tries to find the weights that minimize the difference between the outcomes for the treated object and the synthetic control in the pretreatment period.

The main advantages of the robust model are its ability to handle missing data and its speed. The original model involves a
non-convex optimization which can make parameter estimation take too long, especially when many synthetic controls need to be generated. The robust model,
on the other hand, utilizes a ridge regression with a scalar regularization parameter, which makes it possible to use singular value decomposition for estimation.

The package currently implements most of the optimizations suggested by Amjad et al for their Algorithm 1 (the non-Bayesian variant), with the notable exception of forward-chaining for
hyperparameter selection - the standard train-test split is used instead.

The package is in a very early pre-alpha for now. That is to mean that it works correctly until a commit stops it from doing so until it works
again :)

# Installation

No distributions are provided yet, so installing directly from the repo is the only option:

`pip install git+https://github.com/agoryuno/robust_control`

# Usage

The main function is `get_control()`. This takes a panel matrix, with objects in the rows and timestep observations in columns, and a row index of the object you want to estimate a control for
and returns the timeseries for the synthetic control, original data passed through a denoising filter and the weights of the synthetic control. See below for full details.

## Function `get_control()`

Given the matrix of values `orig_mat` and the row index 
`treated_i`, computes synthetic controls for each combination
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
    which means a single value)

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


## Using the weights:

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

### Note on CUDA and performance:

Unfortunately, PyTorch's implementation of SVD isn't fully parallelizable on CUDA, so running the
estimation on the GPU may actually be slower than on the CPU (see this issue for details: https://github.com/pytorch/pytorch/issues/41306). For that reason CUDA support is disabled by default
but remains an option for the future.
