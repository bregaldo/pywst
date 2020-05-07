# -*- coding: utf-8 -*-

import numpy as np
from .filters import MorletWavelet, GaussianFilter
from .utils import fft, ifft, subsampleFourier, modulus
from .wst import WST

class WSTOp:
    """
    Wavelet Scattering Transform (WST) operator.
    """
    
    def __init__ (self, M, N, J, L = 8, OS = 0, cplx = False):
        """
        

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

        Returns
        -------
        None.

        """
        self.M, self.N, self.J, self.L, self.OS, self.cplx = M, N, J, L, OS, cplx
        self.loadFilters ()
        
    def loadFilters (self):
        """
        Build the set of low pass and bandpass filters that are used for the transform.
        Low pass filters correspond to Gaussian filters.
        Bandpass filters are Morlet wavelets.

        Returns
        -------
        None.

        """
        self.psi = {} # Bandpass filters
        self.phi = {} # Low pass filters
        
        # Filter parameters
        gamma = 4 / self.L  # Aspect ratio
        sigma0 = 0.8        # Std of the envelope
        k0 = 3 * np.pi / 4  # Central wavenumber of the mother wavelet
        
        # Build psi filters
        for j in range (self.J):
            for theta in range (self.L):
                self.psi [j, theta] = {}
                w = MorletWavelet (self.M, self.N, j, (self.L // 2 - theta) * np.pi / self.L, gamma, sigma0, k0).data
                wF = fft (w).real # The imaginary part is null for Morlet wavelets
                for res in range (j + 1):
                    self.psi [j, theta][res] = subsampleFourier (wF, 2 ** res, normalize = True)
            if self.cplx: # We also need rotations for theta in [pi, 2*pi)
                for theta in range (self.L):
                    self.psi [j, theta + self.L] = {}
                    for res in range (j + 1):
                        # Optimization trick for Morlet wavelets
                        self.psi [j, theta + self.L][res] = fft (np.conjugate (ifft (self.psi [j, theta][res]))).real
        
        # Build phi filters
        g = GaussianFilter (self.M, self.N, self.J - 1, sigma0 = sigma0).data
        gF = np.real (fft (g)) # The imaginary part is null for Gaussian filters
        for res in range (self.J):
            self.phi [res] = subsampleFourier (gF, 2 ** res, normalize = True)
    
    def apply (self, data, local = False):
        """
        Compute the WST of input data.
        
        Input data can be either an image (2D array), or a batch of images (3D arrays, the first dimension is the image index).

        Parameters
        ----------
        data : array
            Input data. 2D (one image) or 3D array (batch of images).
        local : bool, optional
            Do we need local coefficients? The default is False.

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
        
        # Check if we are dealing with a batch of images or not.
        if not (data.ndim == 2 or data.ndim == 3):
            raise Exception ("Bad input shape.")
        
        if np.isrealobj (data):  # We make sure that data do not contain any complex value
            locCplx = False # We do not need the whole set of WST coefficients for real data

        # Output lists
        S = []
        SIndex = [] # Index of the coefficients of shape (5, nbCoeffs). Axis 0 order corresponds to: Layer, j1, theta1, j2, theta2
        
        # Zeroth layer
        U0 = fft (data)
        resU0 = 0
        if local:
            S0 = ifft (subsampleFourier (U0 * phi [0], 2 ** max (J - OS, 0)))
        else:
            S0 = ifft (U0)
        if locCplx:
            S0 = modulus (S0)
        else:
            S0 = S0.real # Discard the null imaginary part
        if local:
            S.append (S0)
        else:
            S.append (S0.mean (axis = (-1, -2)))
        SIndex.append ([0, 0, 0, 0, 0])
    
        # First layer
        for j1 in range (J):
            for theta1 in range ((locCplx + 1) * L): # Note the trick to produce 2 when locCplx == True, and 1 otherwise
                U1 = subsampleFourier (U0 * psi [j1, theta1][resU0], 2 ** max (j1 - resU0 - OS, 0))
                U1 = fft (modulus (ifft (U1)))
                resU1 = resU0 + max (j1 - resU0 - OS, 0)
                
                if local:
                    S1 = ifft (subsampleFourier (U1 * phi [resU1], 2 ** (max (J - resU1 - OS, 0)))).real
                    S.append (S1)
                else:
                    S1 = ifft (U1).real
                    S.append (S1.mean (axis = (-1, -2)))
                SIndex.append ([1, j1, theta1, 0, 0])
            
                # Second layer
                for j2 in range (j1 + 1, J):
                    for theta2 in range (L):
                        U2 = subsampleFourier (U1 * psi [j2, theta2][resU1], 2 ** max (j2 - resU1 - OS, 0))
                        U2 = fft (modulus (ifft (U2)))
                        resU2 = resU1 + max (j2 - resU1 - OS, 0)
                        
                        if local:
                            S2 = ifft (subsampleFourier (U2 * phi [resU2], 2 ** (max (J - resU2 - OS, 0)))).real
                            S.append (S2)
                        else:
                            S2 = ifft (U2).real
                            S.append (S2.mean (axis = (-1, -2)))
                        SIndex.append ([2, j1, theta1, j2, theta2])
                        
        return WST (J, L, np.array (S), index = np.array (SIndex).T, cplx = locCplx)