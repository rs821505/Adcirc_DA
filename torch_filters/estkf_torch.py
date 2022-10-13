import torch
import numpy as np
import BaseFilterTorch


class ESTKFT(BaseFilterTorch):
    """
    Error-subspace transform Kalman Filter
    
    Pytorch Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9
    """


    def _assimilate(self):
        """
        Assimilate Data:
        obtain model dimensions, state and obseervation statistics,
        and run forecasting and analysis steps
        """

        self._get_shapes()
        xfp, _, xbar, _ = self._means()

        wa = self._forecast()
        xa = self._analysis(xbar,xfp,wa)

        return xa.numpy()

    def _projection_matrix(self):
        """
        Create projection matrix:
        create matrix of shape Ne x Ne-1 filled with off diagonal values
        fill diagonal with diagonal values then replace values of last row
        """

        inv_ne = -1/torch.sqrt(self.ne)
        off_diag = -1/(self.ne*(-inv_ne+1))
        diag = 1+off_diag

        a = torch.ones((self.ne,self.ne-1),dtype=torch.float64)*off_diag
        a.fill_diagonal_(diag)
        a[-1,:] = inv_ne
        return a

    def _forecast(self):
        """
        Forecast Step
        """

        a = self._projection_matrix()                            # get projection matrix
        d = torch.sub(self.y,self.hxf.mean(axis=1))              # innovation vector

        hl = self.hxf.matmul(a)
        b1 = torch.diag(1/self.r).matmul(hl)
        c1 = (self.ne-1)*torch.eye(self.ne-1)
        c2 = self.inf_fact*c1 + hl.t().matmul(b1)
        
        eigs,u = torch.linalg.eigh(c2)                                  # evd of c2, assumed symmetric

        d1 = b1.t().matmul(d)
        d2 = u.t().matmul(d1)
        d3 = d2/eigs
        t = u.matmul(torch.diag(1/torch.sqrt(eigs)).matmul(u.t()))   # symmetric square root matrix
        
        wm = u.matmul(d3)                                            # mean weight
        wp = t.matmul(a.t()*torch.sqrt((self.ne-1)))                 # perturbation weight
        w = wm.unsqueeze_(1).add(wp)                                 # total weight matrix + projection matrix transform
        wa = a.matmul(w)                                               

        return wa

    def _analysis(self,xbar,xfp, wa):
        """
        Update Step:
        returns: xa: ensemble analysis 
        """
        return xbar.add(xfp.matmul(wa))
