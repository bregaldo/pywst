# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np


class RWSTModelBase (ABC):
    """
    Base class for the definition of a RWST model, i.e. a model of the angular dependencies of the WST coefficients.
    """

    def __init__(self, L):
        """
        Constructor.

        Parameters
        ----------
        L : int
            Number of angles between 0 and pi.

        Returns
        -------
        None.

        """
        self.L = L
        self.model = self.__class__.__name__
        self.layer1_names = []
        self.layer2_names = []
        self.layer1_pltparams = []
        self.layer2_pltparams = []
        self.layer1_nbparams = len(self.layer1_names)
        self.layer2_nbparams = len(self.layer2_names)
        
    def __str__(self):
        """
        String description of the model.

        Returns
        -------
        None.

        """
        s = "Model: " + type(self).__name__ + "\n"
        s += "--> Layer 0 coefficient: S0\n"
        s += "--> Layer 1 coefficients: "
        for name in self.layer1_names:
            s += name + " - "
        s = s[:-3] + "\n"
        s += "--> Layer 2 coefficients: "
        for name in self.layer2_names:
            s += name + " - "
        s = s[:-3]
        return s
        
    @abstractmethod
    def layer1(self, theta_vals, *params):
        """
        Model for layer 1 coefficients.

        Parameters
        ----------
        theta_vals : array
            theta_1 values.
        *params : float
            Parameters of the model.

        Returns
        -------
        array
            Predictions of the model.

        """
        theta1 = theta_vals
        pass
    
    @abstractmethod
    def layer2(self, theta_vals, *params):
        """
        Model for layer 2 coefficients.

        Parameters
        ----------
        theta_vals : (array, array)
            (theta_1,theta_2) values.
        *params : float
            Parameters of the model.

        Returns
        -------
        array
            Predictions of the model.
        """
        theta1, theta2 = theta_vals
        pass

    @abstractmethod
    def finalize(self, rwst):
        """
        Potential finalization steps on the RWST coefficients.

        Parameters
        ----------
        rwst : RWST
            RWST object to finalize.

        Returns
        -------
        None.

        """
        pass


class RWSTModel1 (RWSTModelBase):
    """
    Usual RWST model (without additional terms).
    """
    
    def __init__(self, L):
        super().__init__(L)
        self.layer1_names = ['S1Iso', 'S1Aniso', 'ThetaRef1']
        self.layer2_names = ['S2Iso1', 'S2Iso2', 'S2Aniso1', 'S2Aniso2', 'ThetaRef2']
        self.layer1_pltparams = [(r'$\hat{S}_1^{\mathrm{iso}}(j_1)$', False),
                                 (r'$\hat{S}_1^{\mathrm{aniso}}(j_1)$', False),
                                 (r'$\theta^{\mathrm{ref,1}}(j_1)$', True)]
        self.layer2_pltparams = [(r'$\hat{S}_2^{\mathrm{iso,1}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{iso,2}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{aniso,1}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{aniso,2}}(j_1,j_2)$', False),
                                 (r'$\theta^{\mathrm{ref,2}}(j_1,j_2)$', True)]
        self.layer1_nbparams = len(self.layer1_names)
        self.layer2_nbparams = len(self.layer2_names)
        
    def layer1(self, theta_vals, *params):
        theta1 = theta_vals
        return params[0] + params[1] * np.cos(2 * np.pi * (theta1 - params[2]) / self.L)
    
    def layer2(self, theta_vals, *params):
        theta1, theta2 = theta_vals
        return params[0] + params[1] * np.cos(2 * np.pi * (theta1 - theta2) / self.L) \
            + params[2] * np.cos(2 * np.pi * (theta1 - params[4]) / self.L) \
            + params[3] * np.cos(2 * np.pi * (theta2 - params[4]) / self.L)
    
    def finalize(self, rwst):
        locShape = rwst.coeffs['m0'].shape[1:]
        
        ###########
        # Layer 1 #
        ###########
        
        indexS1Aniso = self.layer1_names.index("S1Aniso")
        indexThetaRef1 = self.layer1_names.index("ThetaRef1")
        # Lift degeneracy between S1Aniso and ThetaRef1
        for j1 in range(rwst.j_min, rwst.J):
            filtering = rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso] < 0
            rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso, filtering] = np.abs(rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso, filtering])
            rwst.coeffs['m1'][j1 - rwst.j_min, indexThetaRef1, filtering] += self.L / 2
        # Smooth ThetaRef1 values
        rwst.coeffs['m1'][:, indexThetaRef1] = (rwst.coeffs['m1'][:, indexThetaRef1] + self.L / 2) % self.L - self.L / 2
        for j1 in range(rwst.j_min + 1, rwst.J):
            for locIndex in np.ndindex(locShape):
                if rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] - rwst.coeffs['m1'][(j1 - rwst.j_min - 1, indexThetaRef1) + locIndex] > self.L / 2:
                    rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] -= self.L
                elif rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] - rwst.coeffs['m1'][(j1 - rwst.j_min - 1, indexThetaRef1) + locIndex] < -self.L / 2:
                    rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] += self.L
        
        ###########
        # Layer 2 #
        ###########
        
        indexS2Aniso1 = self.layer2_names.index("S2Aniso1")
        indexS2Aniso2 = self.layer2_names.index("S2Aniso2")
        indexThetaRef2 = self.layer2_names.index("ThetaRef2")
        # Lift degeneracy between S2Aniso1 and ThetaRef2
        for j1 in range(rwst.j_min, rwst.J):
            for j2 in range(j1 + 1, rwst.J):
                filtering = rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1] < 0
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1, filtering] = np.abs(rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1, filtering])
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso2, filtering] = - rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso2, filtering]
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2, filtering] += self.L / 2
        # Smooth ThetaRef2 values
        rwst.coeffs['m2'][:, :, indexThetaRef2] = (rwst.coeffs['m2'][:, :, indexThetaRef2] + self.L / 2) % self.L - self.L / 2
        for j1 in range(rwst.j_min, rwst.J):
            for j2 in range(j1 + 1, rwst.J):
                for locIndex in np.ndindex(locShape):
                    if rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] - rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min - 1, indexThetaRef2) + locIndex] > self.L / 2:
                        rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] -= self.L
                    elif rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] - rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min - 1, indexThetaRef2) + locIndex] < -self.L / 2:
                        rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] += self.L


class RWSTModel2 (RWSTModelBase):
    """
    RWST model that takes into account additional terms such as lattice terms and S2Iso3 term.
    """
    
    def __init__(self, L):
        super().__init__(L)
        self.layer1_names = ['S1Iso', 'S1Aniso', 'ThetaRef1', 'S1Lat1', 'S1Lat2']
        self.layer2_names = ['S2Iso1', 'S2Iso2', 'S2Aniso1', 'S2Aniso2', 'ThetaRef2', 'S2Iso3']
        self.layer1_pltparams = [(r'$\hat{S}_1^{\mathrm{iso}}(j_1)$', False),
                                 (r'$\hat{S}_1^{\mathrm{aniso}}(j_1)$', False),
                                 (r'$\theta^{\mathrm{ref,1}}(j_1)$', True),
                                 (r'$\hat{S}_1^{\mathrm{lat, 1}}(j_1)$', False),
                                 (r'$\hat{S}_1^{\mathrm{lat, 2}}(j_1)$', False)]
        self.layer2_pltparams = [(r'$\hat{S}_2^{\mathrm{iso,1}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{iso,2}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{aniso,1}}(j_1,j_2)$', False),
                                 (r'$\hat{S}_2^{\mathrm{aniso,2}}(j_1,j_2)$', False),
                                 (r'$\theta^{\mathrm{ref,2}}(j_1,j_2)$', True),
                                 (r'$\hat{S}_2^{\mathrm{iso,3}}(j_1,j_2)$', False)]
        self.layer1_nbparams = len(self.layer1_names)
        self.layer2_nbparams = len(self.layer2_names)
        
    def layer1(self, theta_vals, *params):
        theta1 = theta_vals
        return params[0] + params[1] * np.cos(2 * np.pi * (theta1 - params[2]) / self.L) \
            + params[3] * np.cos(4 * np.pi * theta1 / self.L) \
            + params[4] * np.cos(8 * np.pi * theta1 / self.L)
    
    def layer2(self, theta_vals, *params):
        theta1, theta2 = theta_vals
        return params[0] + params[1] * np.cos(2 * np.pi * (theta1 - theta2) / self.L) \
            + params[2] * np.cos(2 * np.pi * (theta1 - params[4]) / self.L) \
            + params[3] * np.cos(2 * np.pi * (theta2 - params[4]) / self.L) \
            + params[5] * np.cos(4 * np.pi * (theta1 - theta2) / self.L)
                
    def finalize(self, rwst):
        locShape = rwst.coeffs['m0'].shape[1:]
        
        ###########
        # Layer 1 #
        ###########
        
        indexS1Aniso = self.layer1_names.index("S1Aniso")
        indexThetaRef1 = self.layer1_names.index("ThetaRef1")
        # Lift degeneracy between S1Aniso and ThetaRef1
        for j1 in range(rwst.j_min, rwst.J):
            filtering = rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso] < 0
            rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso, filtering] = np.abs(rwst.coeffs['m1'][j1 - rwst.j_min, indexS1Aniso, filtering])
            rwst.coeffs['m1'][j1 - rwst.j_min, indexThetaRef1, filtering] += self.L / 2
        # Smooth ThetaRef1 values
        rwst.coeffs['m1'][:, indexThetaRef1] = (rwst.coeffs['m1'][:, indexThetaRef1] + self.L / 2) % self.L - self.L / 2
        for j1 in range(rwst.j_min + 1, rwst.J):
            for locIndex in np.ndindex(locShape):
                if rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] - rwst.coeffs['m1'][(j1 - rwst.j_min - 1, indexThetaRef1) + locIndex] > self.L / 2:
                    rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] -= self.L
                elif rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] - rwst.coeffs['m1'][(j1 - rwst.j_min - 1, indexThetaRef1) + locIndex] < -self.L / 2:
                    rwst.coeffs['m1'][(j1 - rwst.j_min, indexThetaRef1) + locIndex] += self.L
        
        ###########
        # Layer 2 #
        ###########
        
        indexS2Aniso1 = self.layer2_names.index("S2Aniso1")
        indexS2Aniso2 = self.layer2_names.index("S2Aniso2")
        indexThetaRef2 = self.layer2_names.index("ThetaRef2")
        # Lift degeneracy between S2Aniso1 and ThetaRef2
        for j1 in range(rwst.j_min, rwst.J):
            for j2 in range(j1 + 1, rwst.J):
                filtering = rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1] < 0
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1, filtering] = np.abs(rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso1, filtering])
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso2, filtering] = - rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexS2Aniso2, filtering]
                rwst.coeffs['m2'][j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2, filtering] += self.L / 2
        # Smooth ThetaRef2 values
        rwst.coeffs['m2'][:, :, indexThetaRef2] = (rwst.coeffs['m2'][:, :, indexThetaRef2] + self.L / 2) % self.L - self.L / 2
        for j1 in range(rwst.j_min, rwst.J):
            for j2 in range(j1 + 1, rwst.J):
                for locIndex in np.ndindex(locShape):
                    if rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] - rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min - 1, indexThetaRef2) + locIndex] > self.L / 2:
                        rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] -= self.L
                    elif rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] - rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min - 1, indexThetaRef2) + locIndex] < -self.L / 2:
                        rwst.coeffs['m2'][(j1 - rwst.j_min, j2 - rwst.j_min, indexThetaRef2) + locIndex] += self.L
