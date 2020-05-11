# -*- coding: utf-8 -*-

import numpy as np
import scipy.fft

def fft (data):
    """
    Parallel FFT.
    """
    return scipy.fft.fft2 (data, workers = -1) # By default, fft2 applies to the last two axes of data.

def ifft (data):
    """
    Parallel inverse FFT.
    """
    return scipy.fft.ifft2 (data, workers = -1) # By default, ifft2 applies to the last two axes of data.

def subsampleFourier (x, k, normalize = False):
    """
    Periodization in Fourier space. This corresponds to a subsampling in physical space.

    Parameters
    ----------
    x : array
        Input data (2D or 3D arrays depending of a potential batch dimension).
    k : int
        DESCRIPTION.
    normalize : bool, optional
        Do we want to retain the L1 norm of the signal? The default is False.

    Returns
    -------
    out : array
        Subsampled data (also in Fourier space).

    """
    if x.ndim == 2: # Typical case: x is an image
        y = x.reshape (k, x.shape [0] // k, k, x.shape [1] // k)
        if normalize:
            out = y.sum (axis = (0, 2))
        else:
            out = y.mean (axis = (0, 2))
        return out
    elif x.ndim == 3: # x is a batch of images
        y = x.reshape (x.shape [0], k, x.shape [1] // k, k, x.shape [2] // k)
        if normalize:
            out = y.sum (axis = (1, 3))
        else:
            out = y.mean (axis = (1, 3))
        return out
    else:
        raise Exception ("Bad input shape.")

def modulus (x):
    return np.absolute (x)