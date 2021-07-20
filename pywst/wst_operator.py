# -*- coding: utf-8 -*-
# This source file is inspired from the Kymatio project: https://www.kymat.io/index.html

import os
import sys
from functools import partial
import warnings
import numpy as np
import numpy.ma as ma
import multiprocessing as mp
from .filters import MorletWavelet, GaussianFilter
from .utils import fft, ifft, subsample_fourier, modulus
from .wst import WST


# Internal function for the parallel pre-building of the bandpass filters (see WSTOp.load_filters)
def _build_bp_para(theta_list, bp_filter_cls, M, N, j, L, gamma, sigma0, k0):
    ret = []
    for theta in theta_list:
        ret.append(bp_filter_cls(M, N, j, (L // 2 - theta) * np.pi / L, gamma, sigma0, k0).data)
    return ret


class WSTOp:
    """
    Wavelet Scattering Transform (WST) operator.
    """
    
    def __init__(self, M, N, J, L=8, OS=0, cplx=False, lp_filter_cls=GaussianFilter, bp_filter_cls=MorletWavelet, j_min=0):
        """
        Constructor.
        
        M and N must be proportional to 2^J.

        Parameters
        ----------
        M : int
            Height of the input images.
        N : int
            Width of the input images.
        J : int
            Number of dyadic scales.
        L : int, optional
            Number of angles between 0 and pi. The default is 8.
        OS : int, optional
            Oversampling parameter. The default is 0.
        cplx : bool, optional
            Set it to true if the WSTOp instance will ever apply to complex data. This would load in memory the whole set of bandpass filters. The default is False.
        lp_filter_cls : type, optional
            Class corresponding to the low-pass filter. The default is GaussianFilter.
        bp_filter_cls : type, optional
            Class corresponding to the bandpass filter. The default is MorletWavelet.
        j_min : type, optional
            Minimum dyadic scale. The default is 0.

        Returns
        -------
        None.

        """
        if M % 2 ** J != 0 or N % 2 ** J != 0:
            raise Exception("Choose values for M and N that are proportional to 2^J.")
        
        self.M, self.N, self.J, self.L, self.OS, self.cplx, self.j_min = M, N, J, L, OS, cplx, j_min
        self.load_filters(lp_filter_cls, bp_filter_cls)
        
    def load_filters(self, lp_filter_cls, bp_filter_cls):
        """
        Build the set of low pass and bandpass filters that are used for the transform.

        Parameters
        ----------
        lp_filter_cls : type, optional
            Class corresponding to the low-pass filter.
        bp_filter_cls : type, optional
            Class corresponding to the bandpass filter.
            
        Returns
        -------
        None.

        """
        self.psi = {} # Bandpass filters
        self.phi = {} # Low-pass filters
        
        # Filter parameters
        gamma = 4 / self.L  # Aspect ratio
        sigma0 = 0.8        # Std of the envelope
        k0 = 3 * np.pi / 4  # Central wavenumber of the mother wavelet
        
        # Build psi filters
        for j in range(self.j_min, self.J):
            # Parallel pre-build
            build_bp_para_loc = partial(_build_bp_para, bp_filter_cls=bp_filter_cls, M=self.M, N=self.N, j=j, L=self.L, gamma=gamma, sigma0=sigma0, k0=k0)
            nb_processes = os.cpu_count()
            work_list = np.array_split(np.arange(self.L), nb_processes)
            pool = mp.Pool(processes=nb_processes)
            results = pool.map(build_bp_para_loc, work_list)
            bp_filters = []
            for i in range(len(results)):
                bp_filters += results[i]
            pool.close()
            
            for theta in range(self.L):
                self.psi[j, theta] = {}
                w = bp_filters[theta]
                wF = fft(w).real # The imaginary part is null for Morlet wavelets
                for res in range(max(j + 1, 1)):
                    self.psi[j, theta][res] = subsample_fourier(wF, 2 ** res, normalize=True, aa_filter=True)
            if self.cplx: # We also need rotations for theta in [pi, 2*pi)
                for theta in range(self.L):
                    self.psi[j, theta + self.L] = {}
                    for res in range(max(j + 1, 1)):
                        # Optimization trick for Morlet wavelets
                        self.psi[j, theta + self.L][res] = fft(np.conjugate(ifft(self.psi[j, theta][res]))).real
        
        # Build phi filters
        g = lp_filter_cls(self.M, self.N, self.J - 1, sigma0=sigma0).data
        gF = np.real(fft(g)) # The imaginary part is null for Gaussian filters
        for res in range(self.J):
            self.phi[res] = subsample_fourier(gF, 2 ** res, normalize=True, aa_filter=True)
    
    def apply(self, data, local=False, crop=0.0):
        """
        Compute the WST of input data.
        
        Input data can be either an image (2D array), or a batch of images (3D arrays, the first dimension is the image index).

        Parameters
        ----------
        data : array
            Input data. 2D (one image) or 3D array (batch of images).
        local : bool, optional
            Do we need local coefficients? The default is False.
        crop : float, optional
            For non-periodic images, local coefficients at the borders may need to be cropped.
            Width of the cropping in 2^J pixels unit before downsampling (i.e. crop = 1 corresponds to 2^J pixels cropped before downsampling).

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        WST
            The WST object that contains the corresponding WST coefficients.

        """
        
        J = self.J
        L = self.L
        OS = self.OS
        phi = self.phi
        psi = self.psi
        locCplx = self.cplx
        j_min = self.j_min
        
        # Check if we are dealing with a batch of images or not.
        if not (data.ndim == 2 or data.ndim == 3):
            raise Exception("Bad input shape.")
            
        # Check endianness for little-endian systems.
        # If input data is big-endian, we swap the byte order.
        # Should be useless with scipy 1.5.0.
        if sys.byteorder == 'little' and data.dtype.byteorder == '>':
            warnings.warn("Warning! Swapping byte order of input data to avoid FFT error.")
            data = data.byteswap().newbyteorder()
        
        if np.isrealobj(data):  # We make sure that data do not contain any complex value
            locCplx = False # We do not need the whole set of WST coefficients for real data
            
        # Check input images shape
        imgShape = (data.shape[-2], data.shape[-1])
        if imgShape[0] != self.M or imgShape[1] != self.N:
            raise Exception("Inconsistent width and height of the input images with the parameters of this operator.")

        # Output lists
        S = []
        SIndex = [] # Index of the coefficients of shape (5, nbCoeffs). Axis 0 order corresponds to: Layer, j1, theta1, j2, theta2
        
        # Zeroth layer
        U0 = fft(data)
        resU0 = 0
        if local:
            S0 = ifft(subsample_fourier(U0 * phi[0], 2 ** max(J - OS, 0)))
        else:
            S0 = ifft(U0)
        if locCplx:
            S0 = modulus(S0)
        else:
            S0 = S0.real # Discard the null imaginary part
        if local:
            S.append(S0)
        else:
            S.append(S0.mean(axis=(-1, -2)))
        SIndex.append([0, 0, 0, 0, 0])
    
        # First layer
        for j1 in range(j_min, J):
            for theta1 in range((locCplx + 1) * L): # Note the trick to produce 2 when locCplx == True, and 1 otherwise
                U1 = subsample_fourier(U0 * psi[j1, theta1][resU0], 2 ** max(j1 - resU0 - OS, 0))
                U1 = fft(modulus(ifft(U1)))
                resU1 = resU0 + max(j1 - resU0 - OS, 0)
                
                if local:
                    S1 = ifft(subsample_fourier(U1 * phi[resU1], 2 ** (max(J - resU1 - OS, 0)))).real
                    S.append(S1)
                else:
                    S1 = ifft(U1).real
                    S.append(S1.mean(axis=(-1, -2)))
                SIndex.append([1, j1, theta1, 0, 0])
            
                # Second layer
                for j2 in range(j1 + 1, J):
                    for theta2 in range(L):
                        U2 = subsample_fourier(U1 * psi[j2, theta2][resU1], 2 ** max(j2 - resU1 - OS, 0))
                        U2 = fft(modulus(ifft(U2)))
                        resU2 = resU1 + max(j2 - resU1 - OS, 0)
                        
                        if local:
                            S2 = ifft(subsample_fourier(U2 * phi[resU2], 2 ** (max(J - resU2 - OS, 0)))).real
                            S.append(S2)
                        else:
                            S2 = ifft(U2).real
                            S.append(S2.mean(axis=(-1, -2)))
                        SIndex.append([2, j1, theta1, j2, theta2])
                        
        S = np.array(S)
        
        # Cropping of the local coefficients at the border of the input images
        if local and crop != 0.0:
            mask = np.zeros(S.shape, bool)
            for i in range(self.M // 2 ** (J - OS)):
                for j in range(self.N // 2 ** (J - OS)):
                    if i < int(crop * 2 ** OS) or i > self.M // 2 ** (J - OS) - int(crop * 2 ** OS) \
                       or j < int(crop * 2 ** OS) or j > self.N // 2 ** (J - OS) - int(crop * 2 ** OS):
                        mask[..., i, j] = True
            if np.sum(~mask) == 0:
                raise Exception("No valid data remains after cropping.")
            S = ma.MaskedArray(S, mask=mask) # We create a numpy masked array to mask the cropped coefficients.
                        
        return WST(J, L, S, index=np.array(SIndex).T, cplx=locCplx, j_min=j_min)
