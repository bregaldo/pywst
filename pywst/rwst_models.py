# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

class RWSTModelBase (ABC):

    def __init__ (self, L):
        self.L = L
        self.model = self.__class__.__name__
        self.layer1Names = []
        self.layer2Names = []
        self.layer1PlotParams = []
        self.layer2PlotParams = []
        self.nbParamsLayer1 = len (self.layer1Names)
        self.nbParamsLayer2 = len (self.layer2Names)
        
    @abstractmethod
    def layer1 (self, thetaVals, *params):
        theta1 = thetaVals
        pass
    
    @abstractmethod
    def layer2 (self, thetaVals, *params):
        theta1, theta2 = thetaVals
        pass

    @abstractmethod
    def finalize (self, rwst):
        pass

class RWSTModel1 (RWSTModelBase):
    
    def __init__ (self, L):
        super ().__init__ (L)
        self.layer1Names = ['S1Iso', 'S1Aniso', 'ThetaRef1']
        self.layer2Names = ['S2Iso1', 'S2Iso2','S2Aniso1', 'S2Aniso2', 'ThetaRef2']
        self.layer1PlotParams = [(r'$\widehat{S}_1^{\mathrm{iso}}(j_1)$', False),
                                 (r'$\widehat{S}_1^{\mathrm{aniso}}(j_1)$', False),
                                 (r'$\theta^{\mathrm{ref,1}}(j_1)$', True)]
        self.layer2PlotParams = [(r'$\widehat{S}_2^{\mathrm{iso,1}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{iso,2}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{aniso,1}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{aniso,2}}(j_1,j_2)$', False),
                                 (r'$\theta^{\mathrm{ref,2}}(j_1,j_2)$', True)]
        self.nbParamsLayer1 = len (self.layer1Names)
        self.nbParamsLayer2 = len (self.layer2Names)
        
    def layer1 (self, thetaVals, *params):
        theta1 = thetaVals
        return params [0] + params [1] * np.cos (2 * np.pi * (theta1 - params [2]) / self.L)
    
    def layer2 (self, thetaVals, *params):
        theta1, theta2 = thetaVals
        return params [0] + params [1] * np.cos (2 * np.pi * (theta1 - theta2) / self.L) \
                + params [2] * np.cos (2 * np.pi * (theta1 - params [4]) / self.L) \
                + params [3] * np.cos (2 * np.pi * (theta2 - params [4]) / self.L)
    
    def finalize (self, rwst):
        locShape = rwst.coeffs ['m0'].shape [1:]
        
        ###########
        # Layer 1 #
        ###########
        
        indexS1Aniso = self.layer1Names.index ("S1Aniso")
        indexThetaRef1 = self.layer1Names.index ("ThetaRef1")
        # Lift degeneracy between S1Aniso and ThetaRef1
        for j1 in range (rwst.J):
            filtering = rwst.coeffs ['m1'][j1, indexS1Aniso] < 0
            rwst.coeffs ['m1'][j1, indexS1Aniso, filtering] = np.abs (rwst.coeffs ['m1'][j1,indexS1Aniso, filtering])
            rwst.coeffs ['m1'][j1, indexThetaRef1, filtering] += self.L / 2
        # Smooth ThetaRef1 values
        rwst.coeffs ['m1'][:, indexThetaRef1] = (rwst.coeffs ['m1'][:, indexThetaRef1] + self.L / 2) % self.L - self.L / 2
        for j1 in range (1, rwst.J):
            for locIndex in np.ndindex (locShape):
                if rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] - rwst.coeffs ['m1'][(j1 - 1, indexThetaRef1) + locIndex] > self.L / 2:
                    rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] -= self.L
                elif rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] - rwst.coeffs ['m1'][(j1 - 1, indexThetaRef1) + locIndex] < -self.L / 2:
                    rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] += self.L
        
        ###########
        # Layer 2 #
        ###########
        
        indexS2Aniso1 = self.layer2Names.index ("S2Aniso1")
        indexS2Aniso2 = self.layer2Names.index ("S2Aniso2")
        indexThetaRef2 = self.layer2Names.index ("ThetaRef2")
        # Lift degeneracy between S2Aniso1 and ThetaRef2
        for j1 in range (rwst.J):
            for j2 in range (j1 + 1, rwst.J):
                filtering = rwst.coeffs ['m2'][j1, j2, indexS2Aniso1] < 0
                rwst.coeffs ['m2'][j1, j2, indexS2Aniso1, filtering] = np.abs (rwst.coeffs ['m2'][j1, j2, indexS2Aniso1, filtering])
                rwst.coeffs ['m2'][j1, j2, indexS2Aniso2, filtering] = - rwst.coeffs ['m2'][j1, j2, indexS2Aniso2, filtering]
                rwst.coeffs ['m2'][j1, j2, indexThetaRef2, filtering] += self.L / 2
        # Smooth ThetaRef2 values
        rwst.coeffs ['m2'][:, :, indexThetaRef2] = (rwst.coeffs ['m2'][:, :, indexThetaRef2] + self.L / 2) % self.L - self.L / 2
        for j1 in range (rwst.J):
            for j2 in range (j1 + 1, rwst.J):
                for locIndex in np.ndindex (locShape):
                    if rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] - rwst.coeffs ['m2'][(j1, j2 - 1, indexThetaRef2) + locIndex] > self.L / 2:
                        rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] -= self.L
                    elif rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] - rwst.coeffs ['m2'][(j1, j2 - 1, indexThetaRef2) + locIndex] < -self.L / 2:
                        rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] += self.L

class RWSTModel2 (RWSTModelBase):
    
    def __init__ (self, L):
        super ().__init__ (L)
        self.layer1Names = ['S1Iso', 'S1Aniso', 'ThetaRef1', 'S1Lat1', 'S1Lat2']
        self.layer2Names = ['S2Iso1', 'S2Iso2', 'S2Aniso1', 'S2Aniso2', 'ThetaRef2', 'S2Iso3']
        self.layer1PlotParams = [(r'$\widehat{S}_1^{\mathrm{iso}}(j_1)$', False),
                                 (r'$\widehat{S}_1^{\mathrm{aniso}}(j_1)$', False),
                                 (r'$\theta^{\mathrm{ref,1}}(j_1)$', True),
                                 (r'$\widehat{S}_1^{\mathrm{lat, 1}}(j_1)$', False),
                                 (r'$\widehat{S}_1^{\mathrm{lat, 2}}(j_1)$', False)]
        self.layer2PlotParams = [(r'$\widehat{S}_2^{\mathrm{iso,1}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{iso,2}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{aniso,1}}(j_1,j_2)$', False),
                                 (r'$\widehat{S}_2^{\mathrm{aniso,2}}(j_1,j_2)$', False),
                                 (r'$\theta^{\mathrm{ref,2}}(j_1,j_2)$', True),
                                 (r'$\widehat{S}_2^{\mathrm{iso,3}}(j_1,j_2)$', False)]
        self.nbParamsLayer1 = len (self.layer1Names)
        self.nbParamsLayer2 = len (self.layer2Names)
        
    def layer1 (self, thetaVals, *params):
        theta1 = thetaVals
        return params [0] + params [1] * np.cos (2 * np.pi * (theta1 - params [2]) / self.L) \
                + params [3] * np.cos (4 * np.pi * theta1 / self.L) \
                + params [4] * np.cos (8 * np.pi * theta1 / self.L)
    
    def layer2 (self, thetaVals, *params):
        theta1, theta2 = thetaVals
        return params [0] + params [1] * np.cos (2 * np.pi * (theta1 - theta2) / self.L) \
                + params [2] * np.cos (2 * np.pi * (theta1 - params [4]) / self.L) \
                + params [3] * np.cos (2 * np.pi * (theta2 - params [4]) / self.L) \
                + params [5] * np.cos (4 * np.pi * (theta1 - theta2) / self.L)
                
    def finalize (self, rwst):
        locShape = rwst.coeffs ['m0'].shape [1:]
        
        ###########
        # Layer 1 #
        ###########
        
        indexS1Aniso = self.layer1Names.index ("S1Aniso")
        indexThetaRef1 = self.layer1Names.index ("ThetaRef1")
        # Lift degeneracy between S1Aniso and ThetaRef1
        for j1 in range (rwst.J):
            filtering = rwst.coeffs ['m1'][j1, indexS1Aniso] < 0
            rwst.coeffs ['m1'][j1, indexS1Aniso, filtering] = np.abs (rwst.coeffs ['m1'][j1,indexS1Aniso, filtering])
            rwst.coeffs ['m1'][j1, indexThetaRef1, filtering] += self.L / 2
        # Smooth ThetaRef1 values
        rwst.coeffs ['m1'][:, indexThetaRef1] = (rwst.coeffs ['m1'][:, indexThetaRef1] + self.L / 2) % self.L - self.L / 2
        for j1 in range (1, rwst.J):
            for locIndex in np.ndindex (locShape):
                if rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] - rwst.coeffs ['m1'][(j1 - 1, indexThetaRef1) + locIndex] > self.L / 2:
                    rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] -= self.L
                elif rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] - rwst.coeffs ['m1'][(j1 - 1, indexThetaRef1) + locIndex] < -self.L / 2:
                    rwst.coeffs ['m1'][(j1, indexThetaRef1) + locIndex] += self.L
        
        ###########
        # Layer 2 #
        ###########
        
        indexS2Aniso1 = self.layer2Names.index ("S2Aniso1")
        indexS2Aniso2 = self.layer2Names.index ("S2Aniso2")
        indexThetaRef2 = self.layer2Names.index ("ThetaRef2")
        # Lift degeneracy between S2Aniso1 and ThetaRef2
        for j1 in range (rwst.J):
            for j2 in range (j1 + 1, rwst.J):
                filtering = rwst.coeffs ['m2'][j1, j2, indexS2Aniso1] < 0
                rwst.coeffs ['m2'][j1, j2, indexS2Aniso1, filtering] = np.abs (rwst.coeffs ['m2'][j1, j2, indexS2Aniso1, filtering])
                rwst.coeffs ['m2'][j1, j2, indexS2Aniso2, filtering] = - rwst.coeffs ['m2'][j1, j2, indexS2Aniso2, filtering]
                rwst.coeffs ['m2'][j1, j2, indexThetaRef2, filtering] += self.L / 2
        # Smooth ThetaRef2 values
        rwst.coeffs ['m2'][:, :, indexThetaRef2] = (rwst.coeffs ['m2'][:, :, indexThetaRef2] + self.L / 2) % self.L - self.L / 2
        for j1 in range (rwst.J):
            for j2 in range (j1 + 1, rwst.J):
                for locIndex in np.ndindex (locShape):
                    if rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] - rwst.coeffs ['m2'][(j1, j2 - 1, indexThetaRef2) + locIndex] > self.L / 2:
                        rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] -= self.L
                    elif rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] - rwst.coeffs ['m2'][(j1, j2 - 1, indexThetaRef2) + locIndex] < -self.L / 2:
                        rwst.coeffs ['m2'][(j1, j2, indexThetaRef2) + locIndex] += self.L
