import torch
import numpy as np

from k_filters.filter_torch import BaseFilterTorch

class EnSRFT(BaseFilterTorch):
    """
    Ensemble Square Root Filter
    """

    def _assimilate(self):
        """
        Assimilate Data
        obtain model dimensions, state and obseervation statistics,
        and run forecasting and analysis steps

        returns: xa: ensemble analysis 
        """
        self._get_shapes()
        rmat = self._obs_error_mat()
        xfp, hxp, xbar, ybar = self._means()

        w = self._forecast(hxp,rmat,ybar)
        xa = self._analysis(xbar,xfp,w)

        return xa.numpy()

    
    def _forecast(self,hxp,rmat,ybar):
        """
        Forecast Step
        """

        i1 = hxp.matmul(hxp.t())                                    # gram matrix of perturbations
        i2 = i1+(self.ne-1)*rmat

        eigs, ev = torch.linalg.eigh(i2)                           # compute eigenvalues and eigenvectors (use that matrix is symmetric and real)

        g1 = ev.matmul(torch.diag(torch.sqrt(1/eigs)) )
        g2 = hxp.t().matmul(g1)

        u,s,_= torch.linalg.svd(g2)
        
     
        rad = torch.ones(self.ne).sub(torch.square(s))    # Compute  sqrt of matrix,
        rad.type(torch.complex64)
        a = torch.diag(torch.sqrt(rad))

        w1p = u.matmul(a)
        w2p = w1p.matmul(u.t())
        d = self.y.sub(ybar.squeeze_(1))                                          # innovation vector

        w1 = ev.t().matmul(d)
        w2 = torch.diag(1/eigs).t().matmul(w1)
        w3 = ev.matmul(w2)
        w4 = hxp.t().matmul(w3)
        w = w2p.add(w4.unsqueeze_(1))

        return w

    def _analysis(self,xbar,xfp,w):
        """
        Update Step:
        returns: xa: ensemble analysis 
        """
        return xbar.add(xfp).matmul(w)