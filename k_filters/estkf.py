import numpy as np
from k_filters import BaseFilter

class ESTKF(BaseFilter):
    """
    Error-subspace transform Kalman Filter
    
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9
    - Analysis ensemble (N_x, N_e)
    """
    
    def _assimilate(self):
        """
        Assimilate Data
        """

        xfp, _, xbar, _ = self._means()

        Wa = self._forecast()
        xa = self._analysis(xbar,xfp,Wa)

        return xa
    
    def _projection_matrix(self):
        """
        Create projection matrix:
        create matrix of shape Ne x Ne-1 filled with off diagonal values
        fill diagonal with diagonal values then replace values of last row
        """

        sqr_ne=-1/np.sqrt(self.Ne)
        off_diag=-1/(self.Ne*(-sqr_ne+1))
        diag=1+off_diag

        A=np.ones((self.Ne,self.Ne-1))*off_diag
        np.fill_diagonal(A,diag)
        A[-1,:]=sqr_ne

        return A
 
    def _forecast(self):
        """
        Forecast Step
        """

        A = self._projection_matrix()                   # get projection matrix

        d=self.y-self.hxf.mean(axis=1)                 # innovation vector

        # error in pseudocode, replace L by A
        HL=self.hxf.dot(A)
        B1=np.diag(1/self.R).dot(HL)
        C1=(self.Ne-1)*np.identity(self.Ne-1)
        C2=C1+HL.T.dot(B1)
        
        #EVD of C2, assumed symmetric
        eigs,U=np.linalg.eigh(C2)
        
        d1=B1.T.dot(d)
        d2=U.T.dot(d1)
        d3=d2/eigs
        T=U.dot(np.diag(1/np.sqrt(eigs)).dot(U.T))       # symmetric square root
        
        wm=U.dot(d3)                                     # mean weight
        Wp=T.dot(A.T*np.sqrt((self.Ne-1)))               # perturbation weight
        W=wm[:,None]+Wp                                  # total weight matrix + projection matrix transform
        Wa = A.dot(W)                  

        return Wa

    def _analysis(self,xbar,xfp, Wa):
        """
        Update Step
        """
        Xa = xbar[:,None] + xfp.dot(Wa)   #Analysis ensemble

        return Xa

