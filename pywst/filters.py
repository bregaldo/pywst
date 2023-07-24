# -*- coding: utf-8 -*-
# This source file is inspired from the Kymatio project: https://www.kymat.io/index.html

import numpy as np
from scipy import ndimage


class Filter:
    """
        Base class for filters.
    """
    
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.data = np.zeros((M, N), complex)
        self.type = self.__class__.__name__
        
        
class GaborFilter (Filter):
    """
    Gabor filter.
    
    We make sure that the gabor filters have an (approximate?) unit L1-norm.
    """
    
    def __init__(self, M, N, j, theta, gamma=1.0, sigma0=1.0, k0=2*np.pi):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float
            Rotation angle.
        gamma : float, optional
            Aspect ratio of the envelope. The default is 1.0.
        sigma0 : float, optional
            Standard deviation of the envelope before its dilation. The default is 1.0.
        k0 : float, optional
            Central wavenumber before the dilation. The default is 2 * np.pi.

        Returns
        -------
        None.

        """
        super().__init__(M, N)
        self.j = j
        self.theta = theta
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma = self.sigma0 * 2 ** j
        self.k0 = k0
        self.k = self.k0 / 2 ** j
        self.build()
        
    def build(self):
        """
        Build the filter for a given set of parameters.
        """
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        RInv = np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]])
        D = np.array([[1, 0], [0, self.gamma ** 2]])
        curv = np.dot(R, np.dot(D, RInv)) / (2 * self.sigma ** 2)
        
        for ex in [-2, -1, 0, 1]:
            for ey in [-2, -1, 0, 1]:
                [xx, yy] = np.mgrid[ex * self.M:self.M + ex * self.M, ey * self.N:self.N + ey * self.N]
                arg = -(curv[0, 0] * xx ** 2 + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy ** 2) + 1.j * (xx * self.k * np.cos(self.theta) + yy * self.k * np.sin(self.theta))
                self.data += np.exp(arg)
                
        normFactor = 2 * np.pi * self.sigma ** 2 / self.gamma
        self.data /= normFactor
        
        
class GaussianFilter (Filter):
    """
    Gaussian filter.
    """
    
    def __init__(self, M, N, j, theta=0.0, gamma=1.0, sigma0=1.0):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
        gamma : float, optional
            Aspect ratio of the envelope. The default is 1.0.
        sigma0 : float, optional
            Standard deviation of the envelope before its dilation. The default is 1.0.

        Returns
        -------
        None.

        """
        super().__init__(M, N)
        self.data = np.zeros((M, N)) # No need for a complex data type
        self.j = j
        self.theta = theta
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma = self.sigma0 * 2 ** j
        self.build()
        
    def build(self):
        """
        Build the filter for a given set of parameters.
        """
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        RInv = np.array([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]])
        D = np.array([[1, 0], [0, self.gamma ** 2]])
        curv = np.dot(R, np.dot(D, RInv)) / (2 * self.sigma ** 2)

        for ex in [-2, -1, 0, 1]:
            for ey in [-2, -1, 0, 1]:
                [xx, yy] = np.mgrid[ex * self.M:self.M + ex * self.M, ey * self.N:self.N + ey * self.N]
                arg = -(curv[0, 0] * xx ** 2 + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy ** 2)
                self.data += np.exp(arg)
                
        normFactor = 2 * np.pi * self.sigma ** 2 / self.gamma
        self.data /= normFactor
        
        
class TopHatFilter (Filter):
    """
    Top-hat filter.
    """
    
    def __init__(self, M, N, j, theta=0.0, gamma=1.0, sigma0=1.0):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float, optional
            Rotation angle. The default is 0.0.
        gamma : float, optional
            Aspect ratio of the envelope. The default is 1.0.
        sigma0 : float, optional
            Standard deviation of the envelope before its dilation. The default is 1.0.

        Returns
        -------
        None.

        """
        super().__init__(M, N)
        self.data = np.zeros((M, N)) # No need for a complex data type
        self.j = j
        self.theta = theta
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma = self.sigma0 * 2 ** j
        self.build()
        
    def build(self):
        """
        Build the filter for a given set of parameters.
        """
        
        [xx, yy] = np.mgrid[0:self.M, 0:self.N]
        self.data = ((xx - self.M // 2) ** 2 + ((yy - self.N // 2) * self.gamma) ** 2 <= self.sigma ** 2).astype(float)
        
        self.data = np.fft.fftshift(ndimage.rotate(self.data, self.theta / np.pi * 180.0))
                
        self.data /= np.sum(self.data)
        
        
class Wavelet (Filter):
    """
    Base class for wavelets.
    """
    
    def __init__(self, M, N, j, theta):
        super().__init__(M, N)
        self.j = j
        self.theta = theta
        
        
class MorletWavelet (Wavelet):
    """
    Morlet Wavelet.
    
    We make sure that the wavelets have zero-mean and an (approximate?) unit L1-norm.
    """
    
    def __init__(self, M, N, j, theta, gamma=1.0, sigma0=1.0, k0=2*np.pi):
        """
        Constructor.

        Parameters
        ----------
        M : int
            Height.
        N : int
            Width.
        j : int
            Dyadic scale index.
        theta : float
            Rotation angle.
        gamma : float, optional
            Aspect ratio of the envelope. The default is 1.0.
        sigma0 : float, optional
            Standard deviation of the envelope before the dilation. The default is 1.0.
        k0 : float, optional
            Central wavenumber before the dilation. The default is 2 * np.pi.

        Returns
        -------
        None.

        """
        super().__init__(M, N, j, theta)
        self.gamma = gamma
        self.sigma0 = sigma0
        self.sigma = self.sigma0 * 2 ** j
        self.k0 = k0
        self.build()
        
    def build(self):
        """
        Build the filter for a given set of parameters.
        """
        gabor = GaborFilter(self.M, self.N, self.j, self.theta, self.gamma, self.sigma0, self.k0)
        gaussian = GaussianFilter(self.M, self.N, self.j, self.theta, self.gamma, self.sigma0)
        K = np.sum(gabor.data) / np.sum(gaussian.data)
        self.data = gabor.data - K * gaussian.data
        
