# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import warnings
from .rwst import RWST


class WST:
    """
    Wavelet Scattering Transform (WST) class contains the output of the WST of an image or a batch of images.
    This contains the corresponding coefficients and helps to handle and to plot these coefficients.
    """
    
    def __init__(self, J, L, coeffs, index=None, cplx=False, j_min=0):
        """
        Constructor.

        Parameters
        ----------
        J : int
            Number of dyadic scales.
        L : int
            Number of angles between 0 and pi.
        coeffs : array
            WST coefficients. 1D, 2D, 3D, 4D arrays are valid depending whether
            we have local coefficients or not, whether we are dealing with the WST
            of a batch of images or not.
        index : array, optional
            Index of the coefficients stored in coeffs. The default is None.
        cplx : bool, optional
            Are these coefficients computed from complex images? The default is False.
        j_min : type, optional
            Minimum dyadic scale. The default is 0.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.J = J
        self.L = L
        self.cplx = cplx
        self.j_min = j_min
        
        # Number of coefficients
        self.nb_m0 = 1
        self.nb_m1 = (J - j_min) * L * (1 + cplx)
        self.nb_m2 = (J - j_min) * (J - j_min - 1) * L * L * (1 + cplx) // 2
        
        self.coeffs = coeffs
        self.coeffs_cov = None
        
        # Check if we are dealing with a batch of images coefficients or not, and if we have local coefficients or not.
        if coeffs.ndim == 1:
            self.batch = False
            self.local = False
        elif coeffs.ndim == 2:
            self.batch = True
            self.local = False
        elif coeffs.ndim == 3:
            self.batch = False
            self.local = True
        elif coeffs.ndim == 4:
            self.batch = True
            self.local = True
        else:
            raise Exception("Bad input shape.")
        
        # Reordering and definition of the current index
        if index is not None:
            self.reorder(index)
        self.index = np.zeros((5, self.nb_m0 + self.nb_m1 + self.nb_m2))
        cnt = 1
        for j1 in range(j_min, J):
            for theta1 in range(L * (1 + cplx)):
                self.index[:, cnt] = [1, j1, theta1, 0, 0]
                cnt += 1
        for j1 in range(j_min, J):
            for theta1 in range(L * (1 + cplx)):
                for j2 in range(j1 + 1, J):
                    for theta2 in range(L):
                        self.index[:, cnt] = [2, j1, theta1, j2, theta2]
                        cnt += 1
                        
        # Normalization initialization
        self.normalized = False
        self.normalize_layer1 = None
        self.log2vals = False
        
        # Plot design parameters
        self._xticks, self._xticksMinor, self._xticklabels, self._xticklabelsMinor, self._tickAlignment = [], [], [], [], []
        cnt = 1
        for j1 in range(j_min, J):
            self._xticks.append(cnt + L * (1 + cplx) // 2) # Tick at the middle
            self._xticklabels.append("$j_1 = " + str(j1) + "$")
            self._tickAlignment.append(0.0)
            cnt += L * (1 + cplx)
        for j1 in range(j_min, J - 1):
            for theta1 in range(L * (1 + cplx)):
                for j2 in range(j1 + 1, J):
                    self._xticksMinor.append(cnt + L // 2)
                    self._xticklabelsMinor.append("$j_2 = " + str(j2) + "$")
                    self._tickAlignment.append(-0.1)
                    cnt += L
            self._xticks.append(cnt - L * L * (1 + cplx) * (J - j1 - 1) // 2)
            self._xticklabels.append("$j_1 = " + str(j1) + "$")
            self._tickAlignment.append(0.0)
                
    def reorder(self, index):
        """
        Reorder coefficients (self.coeffs) to the standard lexicographical order (j1, theta1, j2, theta2) (ScatNet order).
        Index parameter specifies the current order.

        Parameters
        ----------
        index : array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        coeffsCopy = self.coeffs.copy()
        
        # Lexicographical order
        reorderIndex = np.lexsort(index[::-1, :])

        # Reorder
        for i in range(coeffsCopy.shape[0]):
            self.coeffs[i, ...] = coeffsCopy[reorderIndex[i], ...]
                
    def _filter_args(self, layer=None, j1=None, theta1=None, j2=None, theta2=None):
        """
        Internal function to filter coefficients.

        Parameters
        ----------
        layer : int, optional
            Selection of the layer of coefficients (0, 1 or 2). The default is all layers.
        j1 : int, optional
            Selection of a specific j_1 values. The default is all j_1 values.
        theta1 : int, optional
            Selection of a specific theta_1 values. The default is all theta_1 values.
        j2 : int, optional
            Selection of a specific j_2 values. The default is all j_2 values.
        theta2 : int, optional
            Selection of a specific theta_2 values. The default is all theta_2 values.

        Returns
        -------
        filtering : array
            Boolean array for coefficients selection.

        """
        filtering = np.ones(self.index.shape[1], bool)
        if layer is not None:
            filtering = np.logical_and(filtering, self.index[0] == layer)
        if j1 is not None:
            filtering = np.logical_and(filtering, self.index[1] == j1)
        if theta1 is not None:
            filtering = np.logical_and(filtering, self.index[2] == theta1)
        if j2 is not None:
            filtering = np.logical_and(filtering, self.index[3] == j2)
        if theta2 is not None:
            filtering = np.logical_and(filtering, self.index[4] == theta2)
        return filtering
            
    def get_coeffs(self, layer=None, j1=None, theta1=None, j2=None, theta2=None):
        """
        Return the selected coefficients.

        Parameters
        ----------
        layer : int, optional
            Selection of the layer of coefficients (0, 1 or 2). The default is all layers.
        j1 : int, optional
            Selection of a specific j_1 values. The default is all j_1 values.
        theta1 : int, optional
            Selection of a specific theta_1 values. The default is all theta_1 values.
        j2 : int, optional
            Selection of a specific j_2 values. The default is all j_2 values.
        theta2 : int, optional
            Selection of a specific theta_2 values. The default is all theta_2 values.

        Returns
        -------
        array
            Array of the coefficients.
        array
            Index of the coefficients.

        """
        filtering = self._filter_args(layer=layer, j1=j1, theta1=theta1, j2=j2, theta2=theta2)
        return self.coeffs[filtering, ...], self.index[:, filtering]
    
    def to_log2(self):
        """
        Computation of logarithmic coefficients (binary logarithm).
        
        If coeffs_cov is not None, keep diagonal coefficients and divide them by (coeffs*log(2))^2 to estimate logarithmic errors at first order.

        Returns
        -------
        self

        """
        if not self.log2vals:
            # Compute the relevant covariance matrix, then compute log2 coefficients.
            if self.coeffs_cov is not None:
                warnings.warn("Warning! The covariance matrix has already been computed with linear coefficients. We compute logarithmic errors from diagonal coefficients and discard off-diagonal coefficients.")
                self.coeffs_cov = np.diag(np.diag(self.coeffs_cov)/(self.coeffs*np.log(2))**2)
            self.coeffs = ma.log2(self.coeffs)
            self.log2vals = True
        return self
            
    def to_linear(self):
        """
        Turn binary logarithmic coefficients into linear coefficients.
        
        If coeffs_cov is not None, keep diagonal coefficients and multiply them by (2^coeffs*log(2))^2 to estimate linear errors at first order.

        Returns
        -------
        self

        """
        if self.log2vals:
            # Compute linear coefficients, then compute the relevant covariance matrix.
            self.coeffs = 2 ** self.coeffs
            if self.coeffs_cov is not None:
                warnings.warn("Warning! The covariance matrix has already been computed with logarithmic coefficients. We compute linear errors from diagonal coefficients and discard off-diagonal coefficients.")
                self.coeffs_cov = np.diag(np.diag(self.coeffs_cov)*(self.coeffs*np.log(2))**2)
            self.log2vals = False
        return self
        
    def normalize(self, normalize_layer1=False):
        """
        Normalization of the coefficients.
        
        Layer 0 coefficients are left unchanged.
        Layer 1 coefficients are left unchanged if normalize_layer1 is False (default). Otherwise normalized by layer 0 coefficients (locally if available):
            
        .. math::
            
            \\bar{S}_1(j_1,\\theta_1) = S_1(j_1,\\theta_1)/S_0
            
        Layer 2 coefficients are normalized by the corresponding layer 1 coefficients (locally if available):
            
        .. math::
            
            \\bar{S}_2(j_1,\\theta_1,j_2,\\theta_2) = S_2(j_1,\\theta_1,j_2,\\theta_2)/S_1(j_1,\\theta_1)

        Parameters
        ----------
        normalize_layer1 : bool, optional
            Should layer 1 be also normalized? The default is False.

        Returns
        -------
        self

        """
        if self.coeffs_cov is not None:
            warnings.warn("Warning! The covariance matrix has been computed before the normalization and will not be updated.")
        
        # Case where no normalization was done or layer 1 normalization is asked to change
        if (not self.normalized) or (self.normalize_layer1 == (not normalize_layer1)):
            coeffsCopy = self.coeffs.copy()
            cnt = 1
            # Layer 1
            for j1 in range(self.j_min, self.J):
                # Case where layer 1 was not normalized but is now asked to be normalized
                if ((not self.normalized) or (not self.normalize_layer1)) and normalize_layer1:
                    if self.log2vals:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] -= coeffsCopy[:1, ...]
                    else:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] /= coeffsCopy[:1, ...]
                # Case where normalization was done with layer 1 normalized and layer 1 is asked not to be normalized
                elif self.normalized and self.normalize_layer1 and (not normalize_layer1):
                    if self.log2vals:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] += coeffsCopy[:1, ...]
                    else:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] *= coeffsCopy[:1, ...]
                cnt += self.L * (1 + self.cplx)
            # Layer 2
            if (not self.normalized):
                for j1 in range(self.j_min, self.J):
                    for theta1 in range(self.L * (1 + self.cplx)):
                        index = 1 + (j1 - self.j_min) * self.L * (1 + self.cplx) + theta1
                        if self.log2vals:
                            self.coeffs[cnt:cnt + self.L * (self.J - j1 - 1), ...] -= coeffsCopy[index:index + 1, ...]
                        else:
                            self.coeffs[cnt:cnt + self.L * (self.J - j1 - 1), ...] /= coeffsCopy[index:index + 1, ...]
                        cnt += self.L * (self.J - j1 - 1)
            self.normalized = True
            self.normalize_layer1 = normalize_layer1
        return self
            
    def unnormalize(self):
        """
        Cancel the normalization of the coefficients. See normalize() method.

        Returns
        -------
        self

        """
        if self.coeffs_cov is not None:
            warnings.warn("Warning! The covariance matrix has been computed before the unnormalization and will not be updated.")
        
        if self.normalized:
            cnt = 1
            for j1 in range(self.j_min, self.J):
                if self.normalize_layer1:
                    if self.log2vals:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] += self.coeffs[:1, ...]
                    else:
                        self.coeffs[cnt:cnt + self.L * (1 + self.cplx), ...] *= self.coeffs[:1, ...]
                cnt += self.L * (1 + self.cplx)
            for j1 in range(self.j_min, self.J):
                for theta1 in range(self.L * (1 + self.cplx)):
                    index = 1 + (j1 - self.j_min) * self.L * (1 + self.cplx) + theta1
                    if self.log2vals:
                        self.coeffs[cnt:cnt + self.L * (self.J - j1 - 1), ...] += self.coeffs[index:index + 1, ...]
                    else:
                        self.coeffs[cnt:cnt + self.L * (self.J - j1 - 1), ...] *= self.coeffs[index:index + 1, ...]
                    cnt += self.L * (self.J - j1 - 1)
            self.normalized = False
            self.normalize_layer1 = None
        return self
                
    def average(self):
        """
        Computation of the mean coefficients and of the corresponding covariance matrix.
        
        This method needs batch data or local coefficients to be effective.

        Returns
        -------
        self

        """
        if self.batch or self.local: # We need multiple samples to average
            coeffsCopy = self.coeffs.copy()
            coeffsCopy = coeffsCopy.reshape((coeffsCopy.shape[0], -1))
            
            # How many valid samples do we have? (assumed to be the same for every coefficient)
            samples = 0
            if type(coeffsCopy) == ma.MaskedArray: # Do we have any masked value?
                samples = coeffsCopy[0].count(axis=-1) # We use S0 coefficient as it should be the same for any other coefficient
            else:
                samples = coeffsCopy.shape[-1]
                
            coeffs_cov = (ma.cov(coeffsCopy) / samples).data # Masked array covariance function to properly handle masked coefficients.
            
            self.coeffs = coeffsCopy.mean(axis=-1)
            self.coeffs_cov = coeffs_cov
            self.coeffs_cov_nbsamples = samples # Keep in memory the number of samples used to compute the sample covariance matrix.
            
            self.batch = False
            self.local = False
        return self
        
    def get_coeffs_cov(self, autoremove_offdiag=True, layer=None, j1=None, theta1=None, j2=None, theta2=None):
        """
        Return the covariance matrix corresponding to the selected coefficients.

        Parameters
        ----------
        layer : int, optional
            Selection of the layer of coefficients (0, 1 or 2). The default is all layers.
        j1 : int, optional
            Selection of a specific j_1 values. The default is all j_1 values.
        theta1 : int, optional
            Selection of a specific theta_1 values. The default is all theta_1 values.
        j2 : int, optional
            Selection of a specific j_2 values. The default is all j_2 values.
        theta2 : int, optional
            Selection of a specific theta_2 values. The default is all theta_2 values.
        autoremove_offdiag : bool, optional
            If we need to guarantee the covariance matrix to be definite, we may need to remove off diagonal coefficients.
            This is done when the effective number of samples to compute the sample covariance matrix is strictly lower than the dimension of the matrix plus one.
            The default is True.

        Returns
        -------
        array
            Covariance matrix corresponding to the selected coefficients.
        array
            Index of the selected coefficients.
        """
        filtering = self._filter_args(layer=layer, j1=j1, theta1=theta1, j2=j2, theta2=theta2)
        if self.coeffs_cov is None:
            return None, self.index[:, filtering]
        else:
            dim = np.sum(filtering)
            covM = self.coeffs_cov[np.outer(filtering, filtering)].reshape(dim, dim)
            # If demanded, remove off diagonal coefficients when we have too few samples to get a definite sample covariance matrix.
            if self.coeffs_cov_nbsamples < dim + 1 and autoremove_offdiag: # We typically expect definite sample covariance matrix when coeffs_cov_nbsamples >= dim + 1.
                covM = np.diag(np.diag(covM))
                warnings.warn("Warning! Removing off diagonal coefficients of the sample covariance matrix (only " + str(self.coeffs_cov_nbsamples) + " samples for dimension " + str(dim) + ").")
            return covM, self.index[:, filtering]
        
    def get_coeffs_std(self, layer=None, j1=None, theta1=None, j2=None, theta2=None):
        """
        Return the standard deviations corresponding to the selected coefficients.

        Parameters
        ----------
        layer : int, optional
            Selection of the layer of coefficients (0, 1 or 2). The default is all layers.
        j1 : int, optional
            Selection of a specific j_1 values. The default is all j_1 values.
        theta1 : int, optional
            Selection of a specific theta_1 values. The default is all theta_1 values.
        j2 : int, optional
            Selection of a specific j_2 values. The default is all j_2 values.
        theta2 : int, optional
            Selection of a specific theta_2 values. The default is all theta_2 values.

        Returns
        -------
        array
            Standard deviations corresponding to the selected coefficients.
        array
            Index of the selected coefficients.
        """
        cov, index = self.get_coeffs_cov(layer=layer, j1=j1, theta1=theta1, j2=j2, theta2=theta2)
        if cov is None:
            return None, index
        else:
            return np.sqrt(np.diag(cov)), index
        
    def _plot(self, axis, x, y, ylabel, legend="", err=None, j1ticks=True):
        """
        Internal plot function.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axes object.
        x : array
            x values.
        y : array
            y values.
        ylabel : str
            Label of y-axis.
        legend : str, optional
            Label for the legend. The default is "".
        err : array, optional
            Error on the coefficients. The default is None.
        j1ticks : bool, optional
            Do we want to show ticks for every j_1 value? The default is True.

        Returns
        -------
        None.

        """
        # Plot
        axis.plot(x, y, '-', markersize=2, label=legend)
        if err is not None:
            axis.fill_between(x, y - err, y + err, alpha=0.2)
        
        # Ticks and grid parameters
        axis.set_ylabel(ylabel)
        if j1ticks:
            axis.set_xticks(self._xticks)
            axis.set_xticklabels(self._xticklabels)
        else:
            axis.set_xticks([])
            axis.set_xticklabels([], visible=False)
        axis.set_xticks(self._xticksMinor, minor=True)
        axis.set_xticklabels(self._xticklabelsMinor, minor=True)
        for t, y in zip(axis.get_xticklabels(), self._tickAlignment):
            t.set_y(y)
        axis.grid(False, axis='x')
        axis.grid(False, axis='x', which='minor')
        axis.tick_params(axis='x', which='minor', direction='out', length=5)
        axis.tick_params(axis='x', which='major', bottom=False, top=False)
        axis.set_xlim(x.min() - 1, x.max() + 1)
        axis.margins(0.2)
        
        # Plot separators
        cnt = 1
        for j1 in range(self.j_min, self.J):
            axis.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
            cnt += self.L * (1 + self.cplx)
        for j1 in range(self.j_min, self.J):
            axis.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
            for theta1 in range(self.L * (1 + self.cplx)):
                if theta1 >= 1:
                    axis.axvline(cnt, color='black', alpha=0.2, linestyle='dashed')
                for j2 in range(j1 + 1, self.J):
                    if j2 > j1 + 1:
                        axis.axvline(cnt, color='black', alpha=0.1, linestyle=':')
                    cnt += self.L
        axis.axvline(cnt, color='black', alpha=0.5, linestyle='dashed')
            
        if legend != "":
            axis.legend()
        
    def plot(self, layer=None, j1=None, title="WST", label=""):
        """
        Plot the WST coefficients.
        Default behaviour plots both layer 1 and layer 2 full set of coefficients.
        
        Parameters
        ----------
        layer : int, optional
            Layer of coefficients to plot (1 or 2). The default is None.
        j1 : int, optional
            Selection of coefficients for a specific j_1 value. The default is None.
        title : str, optional
            Title of the figures. The default is "WST".
        label : str, optional
            Label for the legend. The default is "".

        Returns
        -------
        None.

        """
        self.plot_compare([], layer=layer, j1=j1, title=title, labels=[label])
                
    def plot_compare(self, wst_list, layer=None, j1=None, title="WST", labels=[]):
        """
        Plot of the selected set of WST coefficients of the current object next to those from a given list of other WST or RWST objects.
        
        Default behaviour plots both layer 1 and layer 2 full set of coefficients.
        
        Note that the current object and the objects of wst_list must be of consistent J, L, cplx and j_min parameters.

        Parameters
        ----------
        wst_list : (R)WST or list of (R)WST
            WST object or RWST object, or list of multiple WST/RWST objects.
        layer : int, optional
            Layer of coefficients to plot (1 or 2). The default is None.
        j1 : int, optional
            Selection of coefficients for a specific j_1 value. The default is None.
        title : str, optional
            Title of the figures. The default is "WST".
        labels : list of str, optional
            List of labels to identify the current object and the input objects.
            If wst_list is of length N, labels should be of length N+1 at most, where the first element refers to the label of the current object.
            Default label is "".

        Returns
        -------
        None.
        
        """
        # Check input
        if type(wst_list) != list:
            wst_list_loc = [wst_list] # Easier to make a list in the following
        else:
            wst_list_loc = wst_list.copy() # Local shallow copy

        for elt in wst_list_loc:
            if type(elt) != WST and type(elt) != RWST:
                raise Exception("wst_list must be a WST object or a RWST object, or a list of RWST or/and WST objects!")
            else:
                if self.J != elt.J or self.L != elt.L or self.j_min != elt.j_min or (type(elt) == WST and self.cplx != elt.cplx):
                    raise Exception("Inconsistent (R)WST objects.")
                if type(elt) == WST and (self.log2vals != elt.log2vals or self.normalized != elt.normalized):
                    raise warnings.warn("Warning! Input (R)WST objects have not consistent normalizations compared to the current WST object.")
            
        if layer is None:
            self.plot(layer=1, j1=j1, title=title)
            self.plot(layer=2, j1=j1, title=title)
        elif layer == 1 or layer == 2:
            wst_list_loc = [self] + wst_list_loc # Add current object to wst_list_loc
            
            # Check labels
            if len(labels) < len(wst_list_loc):
                labels += [""] * (len(wst_list_loc) - len(labels)) # Fill up labels list to be consistent with the length of wst_list
            
            # Selection of x values
            xValues = np.array(range(self.nb_m0 + self.nb_m1 + self.nb_m2))
            if j1 is None:
                xSelection = self._filter_args(layer=layer)
            else:
                xSelection = self._filter_args(layer=layer, j1=j1)
            
            # Figure/axis creation
            if layer == 1:
                fig = plt.figure()
            elif layer == 2:
                fig = plt.figure(figsize=(30, 5))
            ax = fig.add_subplot(1, 1, 1)
                
            # ylabel design
            ylabel = r"$" + self.log2vals * r"\log_2(" + self.normalized * (layer==2 or self.normalize_layer1==True) * r"\overline{" + r"S" + self.normalized * (layer==2 or self.normalize_layer1==True) * r"}" + r"_" + str(layer) + self.log2vals * r")" + r"$"
            
            # Plot
            for wst_index, wst_curr in enumerate(wst_list_loc):
                
                if type(wst_curr) == RWST:
                    wst_curr = wst_curr.to_wst(cplx=self.cplx) # If RWST object, convert it first to a WST object.
                
                # Get the shape of the local information (batch + local coefficients per map)
                loc_shape = wst_curr.coeffs.shape[1:]
                # Get a mask associated to S2 coefficient if any local coefficient turns out to be masked (we assume uniform mask along coefficient index axis).
                mask = ma.getmaskarray(wst_curr.coeffs[-1])
                # Define an index of local positions that are not masked.
                validLocIndex = [loc for loc in np.ndindex(loc_shape) if mask[loc] == False]
                
                for locIndex in validLocIndex:
                    # Label design
                    label = labels[wst_index]
                    if label != "" and locIndex != ():
                        label += " - "
                    if locIndex != ():
                        label += str(locIndex)
                    
                    coeffs = wst_curr.coeffs[tuple([xSelection]) + locIndex]
                    if wst_curr.coeffs_cov is None:
                        coeffsErr = None
                    else:
                        coeffsErr = np.sqrt(wst_curr.coeffs_cov.diagonal()[tuple([xSelection]) + locIndex])
                            
                    self._plot(ax, xValues[xSelection], coeffs, ylabel, legend=label, err=coeffsErr)
            
            # Finalization
            plt.subplots_adjust(bottom=0.2)
            plt.title(title + " - m = " + str(layer))
            if layer == 2:
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
