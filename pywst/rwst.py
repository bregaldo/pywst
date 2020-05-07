# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class RWST:
    
    def __init__ (self, J, L, model, locShape = ()):
        self.J = J
        self.L = L
        self.model = model
        
        self.coeffs = {}
        self.coeffsCov = {}
        
        # Initialization for each of the three layer of coefficients
        self.coeffs ['m0'] = np.zeros ((1,) + locShape)
        self.coeffs ['m1'] = np.zeros ((J, model.nbParamsLayer1 + 1) + locShape)       # +1 to add chi2r coefficients
        self.coeffs ['m2'] = np.zeros ((J, J, model.nbParamsLayer2 + 1) + locShape)    # +1 to add chi2r coefficients
        self.coeffsCov ['m0'] = np.zeros ((1, 1) + locShape)
        self.coeffsCov ['m1'] = np.zeros ((J, model.nbParamsLayer1, model.nbParamsLayer1) + locShape)
        self.coeffsCov ['m2'] = np.zeros ((J, J, model.nbParamsLayer2, model.nbParamsLayer2) + locShape)

    def setCoeffs (self, layer, jVals, coeffs, coeffsCov, chi2r):
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
            
    def finalize (self):
        self.model.finalize (self)
            
    def _thetaLabels (self, thetaRange):
        """
            thetaRange must be an array of integers
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
                    for locIndex in np.ndindex (locShape):
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
                    for locNum, locIndex in enumerate (np.ndindex (locShape)):
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
