import numpy as np
from k_filters import BaseFilter


class SEnKF(BaseFilter):
    """
    Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    """
 
    def _assimilate(self):
        """
        Assimilate Data
        """

        Rmat = self._obs_error_mat()
        xfp, hxp, _, _ = self._means()

        A, D = self._forecast(hxp, Rmat)
        xa = self._analysis(A,D,xfp,hxp)
        return xa


    def _forecast(self,hxp,Rmat):
        """
        Forecast Step
        """

        HPH= hxp@ hxp.T /(self.Ne-1)

        A= HPH + Rmat

        rng = np.random.default_rng(seed=42)
        y_p=rng.standard_normal((self.Ny, self.Ne))*np.sqrt(self.R)[:,None]

        D= self.y[:,None]+y_p - self.hxf

        return A, D
    
    def _analysis(self, A, D, xfp, hxp):
        """
        Analysis Step
        """
        # solve linear system for getting inverse
        C=np.linalg.solve(A,D)
        E= hxp.T @ C
        Xa=self.xf + xfp@(E/(self.Ne-1))

        return Xa