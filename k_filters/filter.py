import numpy as np



class BaseFilter:
    """
    Base filter class representing a single data assimilation step
    Class contaings methods shared by all filters.
    """

    def __init__(self,xf, hxf, y, R):

        """
        :param xf:  Prior ensemble ( Nx x N_e )
        :param hxf:   Observations from model ( Ny x Ne )
        :param y:  Observations ( N_y x 1)
        :param R: Observation error (uncorrelated, R is assumed diagonal) ( Ny x 1)
        """
        self.xf = xf
        self.hxf = hxf
        self.y = y
        self.R = R

        self._get_shapes()

    def _get_shapes(self):
        self.Ny = np.size(self.y)
        self.Nx, self.Ne = self.xf.shape

    def _means(self):
        """
        returns: Perturbations from ensemble mean, Obs standard error
        """                 
        xbar = self.xf.mean(axis=1)
        ybar = self.hxf.mean(axis=1)
        return self.xf-xbar[:,None],  self.hxf - np.expand_dims(ybar,axis=1), xbar, ybar

    def _obs_error_mat(self): 
        """
        returns: Observation covariance matrix
        """    
        return np.diag(self.R) 