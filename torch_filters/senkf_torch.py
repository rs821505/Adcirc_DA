import torch
import numpy as np
import time
import BaseFilterTorch


class SEnKFT(BaseFilterTorch):
    """
    Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    """
 
    def _assimilate(self):
        """
        Assimilate data
        obtain model dimensions, state and obseervation statistics,
        and run forecasting and analysis steps

        returns: xa: ensemble analysis 
        """
        self._get_shapes()
        rmat = self._obs_error_mat()
        xfp, hxp, _, _ = self._means()

        a, d = self._forecast(hxp, rmat)
        xa = self._analysis(a,d,xfp,hxp)

        return xa.numpy()


    def _forecast(self,hxp,rmat):
        """
        Forecast Step
        """
        torch.manual_seed(42)

        hph= hxp.matmul(hxp.T) /(self.ne-1)

        a = hph.add(rmat)

        y_p = torch.randn((self.ny, self.ne),dtype=torch.float64)*torch.sqrt(self.r).unsqueeze_(1)

        d = self.y.unsqueeze_(1).add(y_p.sub(self.hxf))

        return a, d
    
    def _analysis(self, a, d, xfp, hxp):
        """
        Analysis Step
        returns: xa: ensemble analysis 
        """
        
        c = torch.linalg.solve(a,d)                         # solve linear system for getting inverse
        e = hxp.t().matmul(c)

        return self.xf.add(xfp.matmul((e/(self.ne-1))))
