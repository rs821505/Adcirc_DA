import torch
import numpy as np


class BaseFilterTorch:
    """
    Base filter class representing a single data assimilation step
    Class contaings methods shared by all filters.
    """

    def __init__(self,xf, hxf, y, r):
        """
        Parameters:
        -------------
        :param xf:  prior ensemble or state forcast array ( nx x ne )
        :param hxf: model observations ( ny x ne )
        :param y:   measurements/observations  ( ny x 1)
        :param r:   observation error (uncorrelated, r is assumed diagonal) ( ny x 1)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
        self.xf = self._to_torch(xf.copy())
        self.hxf = self._to_torch(hxf.copy())
        self.y = self._to_torch(y.copy())
        self.r = self._to_torch(r.copy())
        self.inf_fact = torch.tensor(1)


        
    def _to_torch(self, nparray):
        """
        Convert numpy array to torch tensor
        """
        return torch.as_tensor(nparray,device=self.device, dtype=torch.float32)
        
    def _get_shapes(self):
        """
        Dimensions:
        -----------------------
         ne: ensemble size
         nx: State vector size (Gridboxes x assimilated variables)
         ny: Number of observations
        """
        
        self.ne = torch.tensor(self.xf.shape[1])
        self.nx = torch.tensor(self.xf.shape[0])
        self.ny = torch.tensor(self.y.shape[0])
       

    def _means(self):
        """
        returns: xfp: perturbations from ensemble mean
                hxp: observation standard error
                xbar: ensemble mean
                ybar: observations mean
        """                 
        xbar = self.xf.mean(axis=1)
        ybar = self.hxf.mean(axis=1)
        return self.xf.sub(xbar.unsqueeze_(1)),  self.hxf.sub(ybar.unsqueeze_(1)), xbar, ybar

    def _obs_error_mat(self): 
        """
        returns: r: observation covariance matrix
        """    
        return torch.diag(self.r)