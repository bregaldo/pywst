# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

class RWST:
    """
    Reduced Wavelet Scattering Transform (RWST) class contains the output of the RWST of an image or a batch of images.
    This contains the corresponding coefficients and helps to handle and to plot these coefficients.
    """
    
    def __init__ (self, J, L, model, locShape = ()):
        """
        Constructor.

        Parameters
        ----------
        J : int
            Number of dyadic scales.
        L : int
            Number of angles between 0 and pi.
        model : RWSTModelBase
            Chosen RWST model object.
        locShape : tuple of ints, optional
            Shape of the local dimensions (batch dimension + local dimensions)

        Returns
        -------
        None.

        """
        self.J = J
        self.L = L
        self.model = model
        
        self.coeffs = {}
        self.coeffsCov = {}
        
        # Initialization for each of the three layer of coefficients
        self.coeffs ['m0'] = np.zeros ((1,) + locShape)
        self.coeffs ['m1'] = np.zeros ((J, model.nbParamsLayer1 + 1) + locShape)       # +1 to add chi2r coefficients
        self.coeffs ['m2'] = np.zeros ((J, J, model.nbParamsLayer2 + 1) + locShape)    # +1 to add chi2r coefficients
        self.coeffsCov ['m0'] = np.zeros ((1, 1) + locShape) # Actually, it is not possible yet to get local information for this m0 covariance matrix.
        self.coeffsCov ['m1'] = np.zeros ((J, model.nbParamsLayer1, model.nbParamsLayer1) + locShape)
        self.coeffsCov ['m2'] = np.zeros ((J, J, model.nbParamsLayer2, model.nbParamsLayer2) + locShape)
        
    def _set_mask (self, mask):
        """
        Use MaskedArray objects to deal with masked values when applicable.

        Parameters
        ----------
        mask : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.coeffs ['m0'] = ma.MaskedArray (self.coeffs ['m0'])
        self.coeffs ['m0'][:, mask] = ma.masked
        self.coeffs ['m1'] = ma.MaskedArray (self.coeffs ['m1'])
        self.coeffs ['m1'][:, :, mask] = ma.masked
        self.coeffs ['m2'] = ma.MaskedArray (self.coeffs ['m2'])
        self.coeffs ['m2'][:, :, :, mask] = ma.masked
        self.coeffsCov ['m0'] = ma.MaskedArray (self.coeffsCov ['m0'])
        if self.coeffsCov ['m0'][0, 0].shape == mask.shape: # Special case for m0 rwst coefficient as it is not possible for now to get local information.
            self.coeffsCov ['m0'][:, :, mask] = ma.masked
        self.coeffsCov ['m1'] = ma.MaskedArray (self.coeffsCov ['m1'])
        self.coeffsCov ['m1'][:, :, :, mask] = ma.masked
        self.coeffsCov ['m2'] = ma.MaskedArray (self.coeffsCov ['m2'])
        self.coeffsCov ['m2'][:, :, :, :, mask] = ma.masked

    def _setCoeffs (self, layer, jVals, coeffs, coeffsCov, chi2r):
        """
        Internal functions. Set the values of the coefficients during the optimization (see RWSTOp.apply function).

        Parameters
        ----------
        layer : int
            Layer of coefficients.
        jVals : int or (int,int)
            If layer == 1, j_1 value. If layer == 2, (j_1,j_2) values. Otherwise, ignored.
        coeffs : array
            Coefficients found during the optimization.
        coeffsCov : array
            Covariance matrix corresponding to the coefficients.
        chi2r : TYPE
            Reduced chi square values for the corresponding optimization.

        Returns
        -------
        None.

        """
        if layer == 0:
            self.coeffs ['m0'] = coeffs
            self.coeffsCov ['m0'] = coeffsCov
        elif layer == 1:
            self.coeffs ['m1'][jVals, :-1] = coeffs
            self.coeffs ['m1'][jVals, -1:] = chi2r
            self.coeffsCov ['m1'][jVals] = coeffsCov
        elif layer == 2:
            j1, j2 = jVals
            self.coeffs ['m2'][j1, j2, :-1] = coeffs
            self.coeffs ['m2'][j1, j2, -1:] = chi2r
            self.coeffsCov ['m2'][j1, j2] = coeffsCov

    def getCoeffs (self, name):
        """
        Returns the selected set of RWST coefficients.

        Parameters
        ----------
        name : str
            Can be 'S0' for layer 0 coefficients, 'chi2r1' or 'chi2r2' for reduced chi square values (for layer 1 and layer 2 optimization respectively),
            or the name of a term of the corresponding model (consistent with model.layer1Names and model.layer2Names).

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        array
            Corresponding coefficients.

        """
        if name == 'S0':
            return self.coeffs ['m0']
        elif name == 'chi2r1':
            return self.coeffs ['m1'][:, -1]
        elif name == 'chi2r2':
            return self.coeffs ['m2'][:, :, -1]
        else:
            for index, nameList in enumerate (self.model.layer1Names):
                if name == nameList:
                    return self.coeffs ['m1'][:, index]
            for index, nameList in enumerate (self.model.layer2Names):
                if name == nameList:
                    return self.coeffs ['m2'][:, :, index]
        raise Exception ("Unknown name of parameter: " + str (name))
        
    def getCoeffsStd (self, name):
        """
        Returns the standard deviation of the selected set of RWST coefficients.

        Parameters
        ----------
        name : str
            Can be 'S0' for layer 0 coefficients, or the name of a term of the corresponding model
            (consistent with model.layer1Names and model.layer2Names).

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        array
            Corresponding standard deviation values.

        """
        if name == 'S0':
            return np.sqrt (self.coeffsCov ['m0'][0, 0])
        else:
            for index, nameList in enumerate (self.model.layer1Names):
                if name == nameList:
                    return np.sqrt (self.coeffsCov ['m1'][:, index, index])
            for index, nameList in enumerate (self.model.layer2Names):
                if name == nameList:
                    return np.sqrt (self.coeffsCov ['m2'][:, :, index, index])
        raise Exception ("Unknown name of parameter: " + str (name))
        
    def getCoeffsCov (self, layer = None, j1 = None, j2 = None):
        """
        Returns the covariance matrix of the selected set of RWST coefficients.

        Parameters
        ----------
        layer : int, optional
            Layer of coefficients (must be 0, 1 or 2). The default is None.
        j1 : int, optional
            Specific j_1 value. The default is None.
        j2 : int, optional
            Specific j_2 value. The default is None.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        array
            Corresponding covariance matrix.

        """
        if layer == 0:
            return self.coeffsCov ['m0']
        elif layer == 1:
            if j1 is None:
                return self.coeffsCov ['m1']
            else:
                return self.coeffsCov ['m1'][j1]
        elif layer == 2:
            if j1 is None and j2 is None:
                return self.coeffsCov ['m2']
            elif j1 is None and j2 is not None:
                return self.coeffsCov ['m2'][:, j2]
            elif j1 is not None and j2 is None:
                return self.coeffsCov ['m2'][j1, :]
            else:
                return self.coeffsCov ['m2'][j1, j2]
        else:
            raise Exception ("Choose a layer between 0 and 2!")
            
    def _finalize (self):
        """
        Internal function that is called by RWSTOp.apply.
        Call the corresponding model finalize function that typically deal with potential model parameters degeneracies

        Returns
        -------
        None.

        """
        self.model.finalize (self)
            
    def _thetaLabels (self, thetaRange):
        """
        Internal function. thetaRange must be an array of integers
        """
        ret = []
        for theta in thetaRange:
            s = ""
            if theta == 0:
                s = "$0$"
            elif theta == self.L // 4:
                    s = r"$\frac{\pi}{4}$"
            elif theta == -self.L // 4:
                    s = r"$-\frac{\pi}{4}$"
            elif theta == self.L // 2:
                    s = r"$\frac{\pi}{2}$"
            elif theta == -self.L // 2:
                    s = r"$-\frac{\pi}{2}$"
            elif theta == 3 * (self.L // 4):
                    s = r"$\frac{3\pi}{4}$"
            elif theta == - 3 * (self.L // 4):
                    s = r"$-\frac{3\pi}{4}$"
            elif theta == self.L:
                s = r"$\pi$"
            elif theta == -self.L:
                s = r"$-\pi$"
            ret.append (s)
        return ret
            
    def plot (self, names = []):
        """
        Plot of the selected set of RWST coefficients.
        
        Default behaviour plots layer 1 and layer 2 coefficients, and the reduced chi square values.

        Parameters
        ----------
        names : list of str, optional
            List of coefficients we want to plot on the same figure.
            Can be "chi2r1" or "chi2r2" for reduced chi square values, or names included in model.nbParamsLayer1 and model.nbParamsLayer2 variables.
            The default is [].

        Returns
        -------
        None.

        """
        if names == []:
            self.plot (names = self.model.layer1Names)
            self.plot (names = self.model.layer2Names)
            self.plot (names = ["chi2r1", "chi2r2"])
        else:
            indexLayer1 = []
            indexLayer2 = []
            for name in names:
                if name == "chi2r1":
                    indexLayer1.append (self.model.nbParamsLayer1)
                elif name == "chi2r2":
                    indexLayer2.append (self.model.nbParamsLayer2)
                for index, nameList in enumerate (self.model.layer1Names):
                    if name == nameList: indexLayer1.append (index)
                for index, nameList in enumerate (self.model.layer2Names):
                    if name == nameList: indexLayer2.append (index)
                    
            locShape = self.coeffs ['m0'].shape [1:]
             # Get a mask associated to S0 coefficient if any local coefficient turns out to be masked (we assume uniform mask along coefficient index axis).
            mask = ma.getmaskarray (self.coeffs ['m0'] [0])
            # Define an index of local positions that are not masked.
            validLocIndex = [loc for loc in np.ndindex (locShape) if mask [loc] == False]
            
            colorCycle = plt.rcParams['axes.prop_cycle'].by_key ()['color']
                    
            nRows = int (np.sqrt (len (indexLayer1) + len (indexLayer2)))
            nCols = int (np.ceil ((len (indexLayer1) + len (indexLayer2)) / nRows))
            
            fig, axs = plt.subplots (nRows, nCols, figsize = (nCols * 4, nRows * 3))
            j1Vals = np.arange (self.J)
            for index, pos in enumerate (np.ndindex ((nRows, nCols))):
                # Correction for weird behavior of subplots
                if nCols == 1 and nRows > 1:
                    pos = pos [0]
                elif nRows == 1 and nCols > 1:
                    pos = pos [1]
                elif nRows == 1 and nCols == 1:
                    axs = [axs]
                    pos = pos [0]
                    
                if index < len (indexLayer1):
                    index = indexLayer1 [index]
                    for locIndex in validLocIndex:
                        coeffs = self.coeffs ['m1'][np.index_exp [:, index] + locIndex]
                        if index != self.model.nbParamsLayer1: # No std for chi2r1
                            coeffsStd = np.sqrt (self.coeffsCov ['m1'][np.index_exp [:, index, index] + locIndex])
                        else:
                            coeffsStd = np.zeros (coeffs.shape)
                        axs [pos].plot (j1Vals, coeffs, label = str (locIndex))
                        axs [pos].fill_between (j1Vals, coeffs - coeffsStd, coeffs + coeffsStd, alpha = 0.2)
                    axs [pos].legend ()
                    axs [pos].set_xlabel ("$j_1$")
                    if index != self.model.nbParamsLayer1:
                        axs [pos].set_ylabel (self.model.layer1PlotParams [index][0])
                        if self.model.layer1PlotParams [index][1]: # Readable radian ylabels?
                            angleRange = np.arange ((self.coeffs ['m1'][:, index].ravel ().min () // (self.L / 4)) * self.L // 4, (self.coeffs ['m1'][:, index].ravel ().max () // (self.L / 4) + 2) * self.L // 4, self.L // 4)
                            axs [pos].set_yticks (angleRange)
                            axs [pos].set_yticklabels (self._thetaLabels (angleRange))
                    else:
                        axs [pos].set_ylabel (r"$\chi^{2, \mathrm{S_1}}_\mathrm{r}(j_1)$")
                elif index < len (indexLayer1) + len (indexLayer2):
                    index = indexLayer2 [index - len (indexLayer1)]
                    for locNum, locIndex in enumerate (validLocIndex):
                        color = colorCycle [locNum % len (colorCycle)]
                        coeffs = self.coeffs ['m2'][np.index_exp [:, :, index] + locIndex]
                        if index != self.model.nbParamsLayer2: # No std for chi2r2
                            coeffsStd = np.sqrt (self.coeffsCov ['m2'][np.index_exp [:, :, index, index] + locIndex])
                        else:
                            coeffsStd = np.zeros (coeffs.shape)
                        for j1 in np.arange (self.J - 1):
                            j2Vals = np.arange (j1 + 1, self.J)
                            if j1 != self.J - 2:
                                axs [pos].plot (j2Vals, coeffs [j1, j1 + 1:], label = (j1 == 0) * str (locIndex), color = color)
                                axs [pos].fill_between (j2Vals, coeffs [j1, j1 + 1:] - coeffsStd [j1, j1 + 1:], coeffs [j1, j1 + 1:] + coeffsStd [j1, j1 + 1:], alpha = 0.2, color = color)
                            else:
                                axs [pos].errorbar (j2Vals, coeffs [j1, j1 + 1:], fmt = '.', yerr = coeffsStd [j1, j1 + 1:], color = color)
                    axs [pos].legend ()
                    axs [pos].set_xlabel ("$j_2$")
                    if index != self.model.nbParamsLayer2:
                        axs [pos].set_ylabel (self.model.layer2PlotParams [index][0])
                        if self.model.layer2PlotParams [index][1]: # Readable radian ylabels?
                            angleRange = np.arange ((self.coeffs ['m2'][:, :, index].ravel ().min () // (self.L / 4)) * self.L // 4, (self.coeffs ['m2'][:, :, index].ravel ().max () // (self.L / 4) + 2) * self.L // 4, self.L // 4)
                            axs [pos].set_yticks (angleRange)
                            axs [pos].set_yticklabels (self._thetaLabels (angleRange))
                    else:
                        axs [pos].set_ylabel (r"$\chi^{2, \mathrm{S_2}}_\mathrm{r}(j_1,j_2)$")
            plt.tight_layout ()
            plt.show ()
