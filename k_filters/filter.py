import numpy as np
import os,sys
import math


class BaseFilter:
    """
    Base filter class representing a single data assimilation step
    Class contaings methods shared by all filters.
    """

    def __init__(self,Xf, HXf, Y, R):

        """
        Xf - Prior ensemble ( Nx x N_e )
        HXf -  Observations from model ( Ny x Ne )
        Y -  Observations ( N_y x 1)
        R -  Observation error (uncorrelated, R is assumed diagonal) ( Ny x 1)
        """
        self.Xf = Xf
        self.HXf = HXf
        self.Y = Y
        self.R = R

        self._get_shapes()

    def _get_shapes(self):
        self.Ny = np.size(self.Y)
        self.Nx, self.Ne = self.Xf.shape

    def _means(self):
        """
        returns: Perturbations from ensemble mean, Obs standard error
        """                 
        Xmean = self.Xf.mean(axis=1)
        Ymean = self.HXf.mean(axis=1)
        return self.Xf-Xmean[:,None],  self.HXf - np.expand_dims(Ymean,axis=1), Xmean, Ymean

    def _obs_error_mat(self): 
        """
        returns: Observation covariance matrix
        """    
        return np.diag(self.R) 