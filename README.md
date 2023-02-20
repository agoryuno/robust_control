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

The main entry point is the function `get_control(matrix, i, eta_n=10, mu_n=3, cuda=False)` with arguments:

- `matrix` a $K \times N$ matrix where $K$ is the number of objects and $N$ is the number of observations,
- `i` is the row containing the object you want to estimate a synthetic control for,
- `eta_n` - (optional, default is 10) the number of the ridge regression regularization parameter (often refered to also as $\alpha$ or $\lambda$) values to try, increasing this number may improve the fit but will make estimation take longer,
- `mu_n` - (optional, default is 3) the number of $\mu$ values for the denoising phase to try, same rationale as with $\eta$ applies, except setting this to anything beyond 5 will be mostly useless,
- cuda - (optional, default is False) set this to True to enable GPU acceleration.

The function returns a tuple with two $N \times 1$ vectors: the estimated synthetic control and the original denoised outcome values.

### Note on CUDA support:

Unfortunately, PyTorch's implementation of SVD isn't fully parallelizable on CUDA, so running the
estimation of the GPU may actually be slower than on the CPU (see this issue for details: https://github.com/pytorch/pytorch/issues/41306). For that reason CUDA support is disabled by default
but remains an option for the future.