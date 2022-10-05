import numpy as np
from k_filters import BaseFilter

class EnSRF(BaseFilter):
    """
    Ensemble Square Root Filter
    """

    def _assimilate(self):
        """
        Assimilate Data
        """
        Rmat = self._obs_error_mat()
        xfp, hxp, xbar, ybar = self._means()

        W = self._forecast(hxp,Rmat,ybar)
        xa = self._analysis(xbar,xfp,W)

        return xa

    
    def _forecast(self,hxp,Rmat,ybar):
        """
        Forecast Step
        """

        I1= np.matmul(hxp,hxp.T)                                        #Gram matrix of perturbations
        I2=I1+(self.Ne-1)*Rmat

        eigs, ev = np.linalg.eigh(I2)                                   #compute eigenvalues and eigenvectors (use that matrix is symmetric and real)

        #Error in Pseudocode: Square Root + multiplication order 
        G1= ev.dot( np.diag(np.sqrt(1/eigs)) )
        G2= hxp.T.dot(G1)

        U,s,_=np.linalg.svd(G2)
        
        #Compute  sqrt of matrix,
        rad=(np.ones(self.Ne)-np.square(s)).astype(complex)
        rad=np.sqrt(rad)
        A=np.diag(rad)

        W1p= U.dot(A)
        W2p= W1p.dot(U.T)
        d= np.subtract(self.y,ybar)

        w1= ev.T.dot(d)
        w2= np.diag(1/eigs).T.dot(w1)
        w3= ev.dot(w2)
        w4= hxp.T.dot(w3)
        
        W=W2p+w4[:,None]

        return W

    def _analysis(self,xbar,xfp,W):
        """
        Update Step
        """
        Xa= np.matmul(xbar[:,None]+xfp,W)
        return Xa



