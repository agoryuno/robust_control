# robust_control

A PyTorch implementation of the "robust" synthetic control model proposed by Amjad, Shah and Shen \[[arxiv](https://arxiv.org/abs/1711.06940)\].

The robust synthetic control is an extension of the original synthetic control model, proposed by Abadie, Diamond and Hainmueller (for overview see:
https://www.aeaweb.org/articles?id=10.1257/jel.20191450).

The main advantages of the robust model are its ability to handle missing data and its speed. The original model involves a
non-convex optimization which can make parameter estimation take too long, especially when many synthetic controls need to be generated. The robust model,
on the other hand, utilizes a ridge regression with a scalar regularization parameter, which makes it possible to use singular value decomposition for
estimation.

This package utilizes this fact and implements the model with an SVD in PyTorch, making CUDA acceleration available with a "flip of a switch". It also
implements most of the optimizations suggested by Amjad et al for their Algorithm 1 (the non-Bayesian variant).

The package is in a very early pre-alpha for now. That is to mean that it works correctly until a commit stops it from doing so until it works
again :)

