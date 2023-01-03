# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt
import numpy.linalg as la
import numpy.ma as ma
import warnings

from .wst import WST
from .wst_operator import WSTOp
from .rwst import RWST
from .rwst_models import RWSTModel1


class RWSTOp:
    """
    Reduced Wavelet Scattering Transform (RWST) operator.
    """
    
    def __init__(self, M, N, J, L=8, OS=0, cplx=False, model_cls=RWSTModel1, wst_op=None, j_min=0):
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
        model_cls : type, optional
            Class of the RWST model of the angular dependencies. The default is RWSTModel1.
        wst_op : WSTOp, optional
            WSTOp object with consistent parameters that avoids multiple filters loading.
        j_min : type, optional
            Minimum dyadic scale. The default is 0.

        Returns
        -------
        None.
        """
        self.M, self.N, self.J, self.L, self.OS, self.cplx, self.j_min = M, N, J, L, OS, cplx, j_min
        if wst_op is None:
            self.wst_op = WSTOp(M, N, J, L, OS, cplx, j_min=j_min)
        else:
            if wst_op.M == M and wst_op.N == N and wst_op.J == J and wst_op.L == L and wst_op.OS == OS and wst_op.cplx == cplx and wst_op.j_min == j_min:
                self.wst_op = wst_op
            else:
                warnings.warn("Warning! Loading WSTOp new instance because of wst_op inconsistencies.")
                self.wst_op = WSTOp(M, N, J, L, OS, cplx, j_min=j_min)
        self.model_cls = model_cls
    
    def apply(self, data, local=False, wst_average=False, crop=0.0, diag_cov=True):
        """
        Compute the RWST of input data or from a set of pre-computed WST coefficients.
        
        Input data can be either an image (2D array), or a batch of images (3D arrays, the first dimension is the image index) or a WST object.

        Parameters
        ----------
        data : WST or array
            WST object containing pre-computed WST coefficients or input data.
            2D (one image) or 3D array (batch of images).
        local : bool, optional
            Do we need local WST coefficients? Applicable if data is an image or a batch of images. The default is False.
        wst_average : bool, optional
            WST average before RWST computation? Applicable if data is an image or a batch of images. The default is False.
        crop : float, optional
            For non-periodic images, local coefficients at the borders may need to be cropped. Applicable if data is an image or a batch of images.
            Width of the cropping in 2^J pixels unit before downsampling (i.e. crop = 1 corresponds to 2^J pixels cropped before downsampling).
        diag_cov : bool, optional
            Only use the diagonal coefficients of the covariance matrix for curve_fit.

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
        if type(data) == WST:
            wst = data
            if wst.J != self.J:
                raise Exception("Inconsistent J values between RWST operator and input WST coefficients.")
            if wst.j_min != self.j_min:
                raise Exception("Inconsistent j_min values between RWST operator and input WST coefficients.")
            if not wst.log2vals:
                warnings.warn("Input WST coefficients must have logarithmic values. Auto-computation of logarithmic coefficients...")
                wst.to_log2()
        else:
            wst = self.wst_op.apply(data, local, crop=crop)
            wst.to_log2()
            wst.normalize()
            if wst_average:
                wst.average()
        
        # Get the shape of the local information (batch + local coefficients per map)
        loc_shape = wst.coeffs.shape[1:]
        # Get a mask associated to S2 coefficient if any local coefficient turns out to be masked (we assume uniform mask along coefficient index axis).
        mask = ma.getmaskarray(wst.coeffs[-1])
        # Define an index of local positions that are not masked.
        validLocIndex = [loc for loc in np.ndindex(loc_shape) if mask[loc] == False]
        
        model = self.model_cls(self.L)
        rwst = RWST(self.J, self.L, model, loc_shape=loc_shape, j_min=self.j_min)
        
        # We keep relevant information on the WST object
        rwst.wst_log2vals = wst.log2vals
        rwst.wst_normalized = wst.normalized
        
        # Layer 0 coefficient
        coeffs, coeffsIndex = wst.get_coeffs(layer=0)
        coeffs_cov, _ = wst.get_coeffs_cov(layer=0)
        rwst._set_coeffs(0, None, coeffs, coeffs_cov, None)
        
        ##
        # Note on curve_fit/absolute_sigma parameter:
        # If coeffs_cov is None, we set absolute_sigma to False, otherwise it is True.
        # coeffs_cov absolute values are meaningful, this is why we need absolute_sigma=True,
        # but when coeffs_cov is None, the default value used by curve_fit is an identity matrix,
        # that is why we need absolute_sigma=False in this case.
        ##

        # Layer 1 coefficients
        for j1 in range(self.j_min, self.J):
            coeffs, coeffsIndex = wst.get_coeffs(layer=1, j1=j1)
            coeffs_cov, _ = wst.get_coeffs_cov(layer=1, j1=j1, autoremove_offdiag=not diag_cov)
            if diag_cov and coeffs_cov is not None: # Only keep diagonal coefficients if required
                coeffs_cov = np.diag(np.diag(coeffs_cov))
            theta1Vals = coeffsIndex[2]
            
            paramsOpt = np.zeros((model.layer1_nbparams,) + loc_shape)
            paramsCov = np.zeros((model.layer1_nbparams, model.layer1_nbparams) + loc_shape)
            chi2r = np.zeros(loc_shape)
            for locIndex in validLocIndex: # Enumerate the index values corresponding to the shape loc_shape
                # In the following, np.index_exp helps to concatenate slice() object and tuples.
                # We get the optimized parameters paramsOpt and their corresponding covariance matrix. paramsOpt minimizes the chi squared statistics.
                res = opt.curve_fit(model.layer1, theta1Vals, coeffs[np.index_exp[:] + locIndex],
                                    p0=np.ones(model.layer1_nbparams), sigma=coeffs_cov, absolute_sigma=coeffs_cov is not None, maxfev=5000)
                paramsOpt[np.index_exp[:] + locIndex], paramsCov[np.index_exp[:, :] + locIndex] = res
                x = coeffs[np.index_exp[:] + locIndex] - model.layer1(theta1Vals, *tuple(paramsOpt[np.index_exp[:] + locIndex]))
                if coeffs_cov is not None:
                    chi2r[locIndex] = x.T @ la.inv(coeffs_cov) @ x / (len(x) - model.layer1_nbparams)
                else:
                    chi2r[locIndex] = x.T @ x / (len(x) - model.layer1_nbparams)
            rwst._set_coeffs(1, j1, paramsOpt, paramsCov, chi2r)
            
        # Layer 2 coefficients
        for j1 in range(self.j_min, self.J):
            for j2 in range(j1 + 1, self.J):
                coeffs, coeffsIndex = wst.get_coeffs(layer=2, j1=j1, j2=j2)
                coeffs_cov, _ = wst.get_coeffs_cov(layer=2, j1=j1, j2=j2, autoremove_offdiag=not diag_cov)
                if diag_cov and coeffs_cov is not None: # Only keep diagonal coefficients if required
                    coeffs_cov = np.diag(np.diag(coeffs_cov))
                thetaVals = (coeffsIndex[2], coeffsIndex[4])
                
                paramsOpt = np.zeros((model.layer2_nbparams,) + loc_shape)
                paramsCov = np.zeros((model.layer2_nbparams, model.layer2_nbparams) + loc_shape)
                chi2r = np.zeros(loc_shape)
                for locIndex in validLocIndex: # Enumerate the index values corresponding to the shape loc_shape
                    # In the following, np.index_exp helps to concatenate slice() object and tuples.
                    # We get the optimized parameters paramsOpt and their corresponding covariance matrix. paramsOpt minimizes the chi squared statistics.
                    res = opt.curve_fit(model.layer2, thetaVals, coeffs[np.index_exp[:] + locIndex],
                                        p0=np.ones(model.layer2_nbparams), sigma=coeffs_cov, absolute_sigma=coeffs_cov is not None, maxfev=5000)
                    paramsOpt[np.index_exp[:] + locIndex], paramsCov[np.index_exp[:, :] + locIndex] = res
                    x = coeffs[np.index_exp[:] + locIndex] - model.layer2(thetaVals, *tuple(paramsOpt[np.index_exp[:] + locIndex]))
                    if coeffs_cov is not None:
                        chi2r[locIndex] = x.T @ la.inv(coeffs_cov) @ x / (len(x) - model.layer2_nbparams)
                    else:
                        chi2r[locIndex] = x.T @ x / (len(x) - model.layer1_nbparams)
                rwst._set_coeffs(2, (j1, j2), paramsOpt, paramsCov, chi2r)
        
        rwst._finalize() # Deal with potential model parameters degeneracies
        
        if mask.sum() != 0: # We have at least one masked value
            rwst._set_mask(mask) # Same mask as wst data
        
        return rwst
