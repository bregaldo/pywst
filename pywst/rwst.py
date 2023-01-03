# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pywst as pw


class RWST:
    """
    Reduced Wavelet Scattering Transform (RWST) class contains the output of the RWST of an image or a batch of images.
    This contains the corresponding coefficients and helps to handle and to plot these coefficients.
    """
    
    def __init__(self, J, L, model, loc_shape=(), j_min=0):
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
        loc_shape : tuple of ints, optional
            Shape of the local dimensions (batch dimension + local dimensions)
        j_min : type, optional
            Minimum dyadic scale. The default is 0.

        Returns
        -------
        None.

        """
        self.J = J
        self.L = L
        self.model = model
        self.j_min = j_min
        
        self.coeffs = {}
        self.coeffs_cov = {}
        
        # Information on the original WST coefficients
        self.wst_log2vals = False
        self.wst_normalized = False
        
        # Initialization for each of the three layer of coefficients
        self.coeffs['m0'] = np.zeros((1,) + loc_shape)
        self.coeffs['m1'] = np.zeros((J - j_min, model.layer1_nbparams + 1) + loc_shape)       # +1 to add chi2r coefficients
        self.coeffs['m2'] = np.zeros((J - j_min, J - j_min, model.layer2_nbparams + 1) + loc_shape)    # +1 to add chi2r coefficients
        self.coeffs_cov['m0'] = np.zeros((1, 1) + loc_shape) # Actually, it is not possible yet to get local information for this m0 covariance matrix.
        self.coeffs_cov['m1'] = np.zeros((J - j_min, model.layer1_nbparams, model.layer1_nbparams) + loc_shape)
        self.coeffs_cov['m2'] = np.zeros((J - j_min, J - j_min, model.layer2_nbparams, model.layer2_nbparams) + loc_shape)
        
    def _set_mask(self, mask):
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
        self.coeffs['m0'] = ma.MaskedArray(self.coeffs['m0'])
        self.coeffs['m0'][:, mask] = ma.masked
        self.coeffs['m1'] = ma.MaskedArray(self.coeffs['m1'])
        self.coeffs['m1'][:, :, mask] = ma.masked
        self.coeffs['m2'] = ma.MaskedArray(self.coeffs['m2'])
        self.coeffs['m2'][:, :, :, mask] = ma.masked
        self.coeffs_cov['m0'] = ma.MaskedArray(self.coeffs_cov['m0'])
        if self.coeffs_cov['m0'][0, 0].shape == mask.shape: # Special case for m0 rwst coefficient as it is not possible for now to get local information.
            self.coeffs_cov['m0'][:, :, mask] = ma.masked
        self.coeffs_cov['m1'] = ma.MaskedArray(self.coeffs_cov['m1'])
        self.coeffs_cov['m1'][:, :, :, mask] = ma.masked
        self.coeffs_cov['m2'] = ma.MaskedArray(self.coeffs_cov['m2'])
        self.coeffs_cov['m2'][:, :, :, :, mask] = ma.masked

    def _set_coeffs(self, layer, jVals, coeffs, coeffs_cov, chi2r):
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
        coeffs_cov : array
            Covariance matrix corresponding to the coefficients.
        chi2r : TYPE
            Reduced chi square values for the corresponding optimization.

        Returns
        -------
        None.

        """
        if layer == 0:
            self.coeffs['m0'] = coeffs
            self.coeffs_cov['m0'][:] = coeffs_cov
        elif layer == 1:
            self.coeffs['m1'][jVals - self.j_min, :-1] = coeffs
            self.coeffs['m1'][jVals - self.j_min, -1:] = chi2r
            self.coeffs_cov['m1'][jVals - self.j_min] = coeffs_cov
        elif layer == 2:
            j1, j2 = jVals
            self.coeffs['m2'][j1 - self.j_min, j2 -self.j_min, :-1] = coeffs
            self.coeffs['m2'][j1 - self.j_min, j2 - self.j_min, -1:] = chi2r
            self.coeffs_cov['m2'][j1 - self.j_min, j2 - self.j_min] = coeffs_cov

    def get_coeffs(self, name):
        """
        Returns the selected set of RWST coefficients.

        Parameters
        ----------
        name : str
            Can be 'S0' for layer 0 coefficients, 'chi2r1' or 'chi2r2' for reduced chi square values (for layer 1 and layer 2 optimization respectively),
            or the name of a term of the corresponding model (consistent with model.layer1_names and model.layer2_names).

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
            return self.coeffs['m0']
        elif name == 'chi2r1':
            return self.coeffs['m1'][:, -1]
        elif name == 'chi2r2':
            return self.coeffs['m2'][:, :, -1]
        else:
            for index, nameList in enumerate(self.model.layer1_names):
                if name == nameList:
                    return self.coeffs['m1'][:, index]
            for index, nameList in enumerate(self.model.layer2_names):
                if name == nameList:
                    return self.coeffs['m2'][:, :, index]
        raise Exception("Unknown name of parameter: " + str(name))
        
    def get_coeffs_std(self, name):
        """
        Returns the standard deviation of the selected set of RWST coefficients.

        Parameters
        ----------
        name : str
            Can be 'S0' for layer 0 coefficients, or the name of a term of the corresponding model
            (consistent with model.layer1_names and model.layer2_names).

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
            return np.sqrt(self.coeffs_cov['m0'][0, 0])
        else:
            for index, nameList in enumerate(self.model.layer1_names):
                if name == nameList:
                    return np.sqrt(self.coeffs_cov['m1'][:, index, index])
            for index, nameList in enumerate(self.model.layer2_names):
                if name == nameList:
                    return np.sqrt(self.coeffs_cov['m2'][:, :, index, index])
        raise Exception("Unknown name of parameter: " + str(name))
        
    def get_coeffs_cov(self, layer=None, j1=None, j2=None):
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
            return self.coeffs_cov['m0']
        elif layer == 1:
            if j1 is None:
                return self.coeffs_cov['m1']
            else:
                return self.coeffs_cov['m1'][j1 - self.j_min]
        elif layer == 2:
            if j1 is None and j2 is None:
                return self.coeffs_cov['m2']
            elif j1 is None and j2 is not None:
                return self.coeffs_cov['m2'][:, j2 - self.j_min]
            elif j1 is not None and j2 is None:
                return self.coeffs_cov['m2'][j1 - self.j_min, :]
            else:
                return self.coeffs_cov['m2'][j1 - self.j_min, j2 - self.j_min]
        else:
            raise Exception("Choose a layer between 0 and 2!")
            
    def _finalize(self):
        """
        Internal function that is called by RWSTOp.apply.
        Call the corresponding model finalize function that typically deal with potential model parameters degeneracies

        Returns
        -------
        None.

        """
        self.model.finalize(self)
            
    def _theta_labels(self, theta_range):
        """
        Internal function. theta_range must be an array of integers
        """
        ret = []
        for theta in theta_range:
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
            ret.append(s)
        return ret
            
    def plot(self, names=[], label=""):
        """
        Plot of the selected set of RWST coefficients.
        
        Default behaviour plots layer 1 and layer 2 coefficients, and the reduced chi square values.

        Parameters
        ----------
        names : str or list of str, optional
            List of coefficients we want to plot on the same figure.
            Can be "chi2r1" or "chi2r2" for reduced chi square values, or names included in model.layer1_nbparams and model.layer2_nbparams variables.
            The default is [].
        label : str, optional
            Label for the legend. The default is "".

        Returns
        -------
        None.

        """
        self.plot_compare([], names=names, labels=[label])
            
    def plot_compare(self, rwst_list, names=[], labels=[]):
        """
        Plot of the selected set of RWST coefficients of the current object next to those from a given list of other RWST objects.
        
        Default behaviour plots layer 1 and layer 2 coefficients, and the reduced chi square values.
        
        Note that the current object and the objects of rwst_list must be of consistent J, L, RWST model, j_min.

        Parameters
        ----------
        rwst_list : RWST or list of RWST
            RWST object or list of multiple RWST objects.
        names : str or list of str, optional
            List of coefficients we want to plot on the same figure.
            Can be "chi2r1" or "chi2r2" for reduced chi square values, or names included in model.layer1_nbparams and model.layer2_nbparams variables.
            The default is [].
        labels : list of str, optional
            List of labels to identify the current object and the input objects.
            If rwst_list is of length N, labels should be of length N+1 at most, where the first element refers to the label of the current object.
            Default label is "".

        Returns
        -------
        None.
        
        """
        # Check input
        if type(rwst_list) != list:
            rwst_list_loc = [rwst_list] # Easier to make a list in the following
        else:
            rwst_list_loc = rwst_list.copy() # Local shallow copy
        for elt in rwst_list_loc:
            if type(elt) != RWST:
                raise Exception("rwst_list must be a RWST object or a list of RWST objects!")
            else:
                if self.J != elt.J or self.L != elt.L or self.j_min != elt.j_min or type(self.model) != type(elt.model):
                    raise Exception("Inconsistent RWST objects.")
            
        if names == []:
            self.plot_compare(rwst_list, names=self.model.layer1_names, labels=labels)
            self.plot_compare(rwst_list, names=self.model.layer2_names, labels=labels)
            self.plot_compare(rwst_list, names=["chi2r1", "chi2r2"], labels=labels)
        else:
            rwst_list_loc = [self] + rwst_list_loc # Add current object to rwst_list_loc
            
            # Check labels
            if len(labels) < len(rwst_list_loc):
                labels += [""] * (len(rwst_list_loc) - len(labels)) # Fill up labels list to be consistent with the length of rwst_list
            
            # We first make sure to have a list of names for the following
            if type(names) != list:
                names = [names]
                
            indexLayer1 = []
            indexLayer2 = []
            for name in names:
                if name == "chi2r1":
                    indexLayer1.append(self.model.layer1_nbparams)
                elif name == "chi2r2":
                    indexLayer2.append(self.model.layer2_nbparams)
                for index, nameList in enumerate(self.model.layer1_names):
                    if name == nameList:
                        indexLayer1.append(index)
                for index, nameList in enumerate(self.model.layer2_names):
                    if name == nameList:
                        indexLayer2.append(index)
                        
            nRows = int(np.sqrt(len(indexLayer1) + len(indexLayer2)))
            nCols = int(np.ceil((len(indexLayer1) + len(indexLayer2)) / nRows))
            
            fig, axs = plt.subplots(nRows, nCols, figsize=(nCols * 4, nRows * 3))
            
            colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            for ax_index, pos in enumerate(np.ndindex((nRows, nCols))):
                legend = False # No legend except if there is a label to show in the following
                
                # Correction for weird behavior of subplots
                if nCols == 1 and nRows > 1:
                    pos = pos[0]
                elif nRows == 1 and nCols > 1:
                    pos = pos[1]
                elif nRows == 1 and nCols == 1:
                    axs = [axs]
                    pos = pos[0]
                
                color_cnt = 0 # Count for the color cycle
                
                for rwst_index, rwst_curr in enumerate(rwst_list_loc): # For every RWST object in rwst_list
                
                    # Get local parameters
                    loc_shape = rwst_curr.coeffs['m0'].shape[1:]
                    # Get a mask associated to S2 coefficient if any local coefficient turns out to be masked (we assume uniform mask along coefficient index axis).
                    mask = ma.getmaskarray(rwst_curr.coeffs['m2'][0,0,0])
                    # Define an index of local positions that are not masked.
                    validLocIndex = [loc for loc in np.ndindex(loc_shape) if mask[loc] == False]
                    
                    j1Vals = np.arange(rwst_curr.j_min, rwst_curr.J)
                    
                    if ax_index < len(indexLayer1): # Plot instructions for first order coefficients
                        index = indexLayer1[ax_index]
                        for locIndex in validLocIndex:
                            color = colorCycle[color_cnt % len(colorCycle)]
                            
                            coeffs = rwst_curr.coeffs['m1'][np.index_exp[:, index] + locIndex]
                            if index != rwst_curr.model.layer1_nbparams: # No std for chi2r1
                                coeffsStd = np.sqrt(rwst_curr.coeffs_cov['m1'][np.index_exp[:, index, index] + locIndex])
                            else:
                                coeffsStd = np.zeros(coeffs.shape)
                                
                            # Label design
                            label = labels[rwst_index]
                            if label != "" and locIndex != ():
                                label += " - "
                            if locIndex != ():
                                label += str(locIndex)
                            if label != "": legend = True # Need to display a legend
                                
                            axs[pos].plot(j1Vals, coeffs, label=label, color=color)
                            axs[pos].fill_between(j1Vals, coeffs - coeffsStd, coeffs + coeffsStd, alpha=0.2, color=color)
                            
                            color_cnt += 1

                        axs[pos].set_xlabel("$j_1$")
                        if index != rwst_curr.model.layer1_nbparams:
                            axs[pos].set_ylabel(rwst_curr.model.layer1_pltparams[index][0])
                            if rwst_curr.model.layer1_pltparams[index][1]: # Readable radian ylabels?
                                angleRange = np.arange((rwst_curr.coeffs['m1'][:, index].ravel().min() // (rwst_curr.L / 4)) * rwst_curr.L // 4, (rwst_curr.coeffs['m1'][:, index].ravel().max() // (rwst_curr.L / 4) + 2) * rwst_curr.L // 4, rwst_curr.L // 4)
                                axs[pos].set_yticks(angleRange)
                                axs[pos].set_yticklabels(rwst_curr._theta_labels(angleRange))
                        else:
                            axs[pos].set_ylabel(r"$\chi^{2, \mathrm{S_1}}_\mathrm{r}(j_1)$")
                            
                    elif ax_index < len(indexLayer1) + len(indexLayer2): # Plot instructions for second order coefficients
                        index = indexLayer2[ax_index - len(indexLayer1)]
                        for locIndex in validLocIndex:
                            color = colorCycle[color_cnt % len(colorCycle)]
                            
                            coeffs = rwst_curr.coeffs['m2'][np.index_exp[:, :, index] + locIndex]
                            if index != rwst_curr.model.layer2_nbparams: # No std for chi2r2
                                coeffsStd = np.sqrt(rwst_curr.coeffs_cov['m2'][np.index_exp[:, :, index, index] + locIndex])
                            else:
                                coeffsStd = np.zeros(coeffs.shape)
                            for j1 in np.arange(rwst_curr.j_min, rwst_curr.J - 1):
                                j2Vals = np.arange(j1 + 1, rwst_curr.J)
                                if j1 != rwst_curr.J - 2:
                                    
                                    # Label design
                                    label = ""
                                    if j1 == 0:
                                        label = labels[rwst_index]
                                        if label != "" and locIndex != ():
                                            label += " - "
                                        if locIndex != ():
                                            label += str(locIndex)
                                    if label != "": legend = True # Need to display a legend
                                        
                                    axs[pos].plot(j2Vals, coeffs[j1 - self.j_min, j1 - self.j_min + 1:], label=label, color=color)
                                    axs[pos].fill_between(j2Vals, coeffs[j1 - self.j_min, j1 - self.j_min + 1:] - coeffsStd[j1 - self.j_min,j1 - self.j_min + 1:], \
                                                          coeffs[j1 - self.j_min, j1 -self.j_min + 1:] + coeffsStd[j1 - self.j_min, j1 - self.j_min + 1:], alpha=0.2, color=color)
                                else:
                                    axs[pos].errorbar(j2Vals, coeffs[j1 - self.j_min, j1 - self.j_min + 1:], fmt='.', \
                                                      yerr=coeffsStd[j1 -self.j_min, j1 - self.j_min + 1:], color=color)
                                    
                            color_cnt += 1
                        
                        axs[pos].set_xlabel("$j_2$")
                        if index != rwst_curr.model.layer2_nbparams:
                            axs[pos].set_ylabel(rwst_curr.model.layer2_pltparams[index][0])
                            if rwst_curr.model.layer2_pltparams[index][1]: # Readable radian ylabels?
                                angleRange = np.arange((rwst_curr.coeffs['m2'][:, :, index].ravel().min() // (rwst_curr.L / 4)) * rwst_curr.L // 4, (rwst_curr.coeffs['m2'][:, :, index].ravel().max() // (rwst_curr.L / 4) + 2) * rwst_curr.L // 4, rwst_curr.L // 4)
                                axs[pos].set_yticks(angleRange)
                                axs[pos].set_yticklabels(rwst_curr._theta_labels(angleRange))
                        else:
                            axs[pos].set_ylabel(r"$\chi^{2, \mathrm{S_2}}_\mathrm{r}(j_1,j_2)$")
                        
                if legend:
                    axs[pos].legend()
                        
            plt.tight_layout()
            plt.show()
            
    def to_wst(self, cplx=False):
        """
        Return the corresponding WST object of the current object.

        Returns
        -------
        None.

        """
        # Build a WST index
        wst_index = [[0, 0, 0, 0, 0]] # Layer 0
        for j1 in range(self.j_min, self.J): # Layer 1
            for theta1 in range(self.L * (1 + cplx)):
                wst_index.append([1, j1, theta1, 0, 0])
        for j1 in range(self.j_min, self.J): # Layer 2
            for theta1 in range(self.L * (1 + cplx)):
                for j2 in range(j1 + 1, self.J):
                    for theta2 in range(self.L):
                        wst_index.append([2, j1, theta1, j2, theta2])
        wst_index = np.array(wst_index).T
        
        # Get local shape and mask
        loc_shape = self.coeffs['m0'][0].shape
        mask = ma.getmaskarray(self.coeffs['m2'][0,0,0])
        valid_loc_index = [loc for loc in np.ndindex(loc_shape) if mask[loc] == False]
        
        # Initialization
        S = np.zeros((wst_index.shape[1],) + loc_shape)
        if mask.sum() != 0: # We have at least one masked value
            S = ma.MaskedArray(S)
            S[:, mask] = ma.masked
        
        # S0 coefficients
        S[0] = self.coeffs['m0'][0]
        
        # S1 coefficients
        for j1 in range(self.j_min, self.J):
            filtering = np.logical_and(wst_index[0] == 1, wst_index[1] == j1)
            theta_vals = wst_index[2, filtering]
            coeffs = self.coeffs['m1'][j1 - self.j_min, :-1]
            for loc_index in np.ndindex(loc_shape):
                S[(filtering,) + loc_index] = self.model.layer1(theta_vals, *(coeffs[np.index_exp[:] + loc_index]))
            
        # S2 coefficients
        for j1 in range(self.j_min, self.J):
            for j2 in range(j1 + 1, self.J):
                filtering = np.logical_and(wst_index[1] == j1, wst_index[3] == j2)
                theta_vals = (wst_index[2, filtering], wst_index[4, filtering])
                coeffs = self.coeffs['m2'][j1 - self.j_min, j2 - self.j_min, :-1]
                for loc_index in valid_loc_index:
                    S[(filtering,) + loc_index] = self.model.layer2(theta_vals, *(coeffs[np.index_exp[:] + loc_index]))
                
        # We create the WST object and fill out its attributes
        wst = pw.WST(self.J, self.L, S, index=wst_index, cplx=cplx, j_min=self.j_min)
        wst.normalized = self.wst_normalized
        wst.log2vals = self.wst_log2vals
        
        # TODO: add uncertainties to wst
        
        return wst
