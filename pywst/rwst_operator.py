# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
import numpy.linalg as la

from .wst import WST
from .wst_operator import WSTOp
from .rwst import RWST
from .rwst_models import RWSTModel1

class RWSTOp:
    """
    Reduced Wavelet Scattering Transform (RWST) operator.
    """
    
    def __init__ (self, M, N, J, L = 8, OS = 0, cplx = False, model = RWSTModel1):
        """
        Constructor.
        
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
            Set it to true if the RWSTOp instance will ever apply to complex data. This would load in memory the whole set of bandpass filters. The default is False.
        model : type, optional
            Class of the RWST model of the angular dependencies. The default is RWSTModel1.

        Returns
        -------
        None.
        """
        self.M, self.N, self.J, self.L, self.OS, self.cplx = M, N, J, L, OS, cplx
        self.wstOp = WSTOp (M, N, J, L, OS, cplx)
        self.model = model
    
    def apply (self, data, local = False):
        """
        Compute the RWST of input data or from a set of pre-computed WST coefficients.
        
        Input data can be either an image (2D array), or a batch of images (3D arrays, the first dimension is the image index) or a WST object.

        Parameters
        ----------
        data : WST or array
            WST object containing pre-computed WST coefficients or input data.
            2D (one image) or 3D array (batch of images).
        local : bool, optional
            Do we need local coefficients? The default is False.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        rwst : RWST
            The RWST object that contains the corresponding RWST coefficients.

        """
        # Check if data is made of images or is directly a WST object.
        if type (data) == WST:
            wst = data
            if wst.J != self.J:
                raise Exception ("Inconsistent J values between RWST operator and input WST coefficients.")
        else:
            wst = self.wstOp.apply (data, local)
            wst.normalize (log2 = True)
            if not local:
                wst.average ()
                
        locShape = wst.coeffs.shape [1:] # Shape of the local information (batch + local coefficients per map)
        model = self.model (self.L)
        rwst = RWST (self.J, self.L, model, locShape = locShape)
        
        # Layer 0 coefficient
        coeffs, coeffsIndex = wst.getCoeffs (layer = 0)
        coeffsCov, coeffsIndex = wst.getCoeffsCov (layer = 0)
        rwst._setCoeffs (0, None, coeffs, coeffsCov, None)

        # Layer 1 coefficients
        for j1 in range (self.J):
            coeffs, coeffsIndex = wst.getCoeffs (layer = 1, j1 = j1)
            coeffsCov, coeffsIndex = wst.getCoeffsCov (layer = 1, j1 = j1)
            theta1Vals = coeffsIndex [2]
            
            paramsOpt = np.zeros ((model.nbParamsLayer1,) + locShape)
            paramsCov = np.zeros ((model.nbParamsLayer1,model.nbParamsLayer1) + locShape)
            chi2r = np.zeros (locShape)
            for locIndex in np.ndindex (locShape): # Enumerate the index values corresponding to the shape locShape
                # In the following, np.index_exp helps to concatenate slice() object and tuples.
                # We get the optimized parameters paramsOpt and their corresponding covariance matrix. paramsOpt minimizes the chi squared statistics.
                paramsOpt [np.index_exp [:] + locIndex], paramsCov [np.index_exp [:,:] + locIndex] = opt.curve_fit (model.layer1, theta1Vals, coeffs [np.index_exp [:] + locIndex], p0 = np.ones (model.nbParamsLayer1), sigma = coeffsCov)
                x = coeffs [np.index_exp [:] + locIndex] - model.layer1 (theta1Vals, *tuple (paramsOpt [np.index_exp [:] + locIndex]))
                chi2r [locIndex] = x.T @ la.inv (coeffsCov) @ x  / (len (x) - model.nbParamsLayer1)
            rwst._setCoeffs (1, j1, paramsOpt, paramsCov, chi2r)
            
        # Layer 2 coefficients
        for j1 in range (self.J):
            for j2 in range (j1 + 1, self.J):
                coeffs, coeffsIndex = wst.getCoeffs (layer = 2, j1 = j1, j2 = j2)
                coeffsCov, coeffsIndex = wst.getCoeffsCov (layer = 2, j1 = j1, j2 = j2)
                thetaVals = (coeffsIndex [2], coeffsIndex [4])
                
                paramsOpt = np.zeros ((model.nbParamsLayer2,) + locShape)
                paramsCov = np.zeros ((model.nbParamsLayer2,model.nbParamsLayer2) + locShape)
                chi2r = np.zeros (locShape)
                for locIndex in np.ndindex (locShape): # Enumerate the index values corresponding to the shape locShape
                    # In the following, np.index_exp helps to concatenate slice() object and tuples.
                    # We get the optimized parameters paramsOpt and their corresponding covariance matrix. paramsOpt minimizes the chi squared statistics.
                    paramsOpt [np.index_exp [:] + locIndex], paramsCov [np.index_exp [:,:] + locIndex] = opt.curve_fit (model.layer2, thetaVals, coeffs [np.index_exp [:] + locIndex], p0 = np.ones (model.nbParamsLayer2), sigma = coeffsCov)
                    x = coeffs [np.index_exp [:] + locIndex] - model.layer2 (thetaVals, *tuple (paramsOpt [np.index_exp [:] + locIndex]))
                    chi2r [locIndex] = x.T @ la.inv (coeffsCov) @ x  / (len (x) - model.nbParamsLayer2)
                rwst._setCoeffs (2, (j1, j2), paramsOpt, paramsCov, chi2r)
        
        rwst._finalize () # Deal with potential model parameters degeneracies
        
        return rwst
