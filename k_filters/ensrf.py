import numpy as np
from k_filters import BaseFilter

class EnSRF(BaseFilter):
    """
    Dimensions: N_e: ensemble size,
                N_y: Number of observations:
                N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
    """

    def _assimilate(self):
        """
        Assimilate Data
        """
        Rmat = self._obs_error_mat()
        Xfp, HXp, Xmean, Ymean = self._means()

        W = self._analysis(HXp,Rmat,Ymean)
        xa = self._update(Xmean,Xfp,W)

        return xa

    
    def _analysis(self,HXp,Rmat,Ymean):
        """
        Analysis Step
        """

        I1= np.matmul(HXp,HXp.T)                                        #Gram matrix of perturbations
        I2=I1+(self.Ne-1)*Rmat

        eigs, ev = np.linalg.eigh(I2)                                   #compute eigenvalues and eigenvectors (use that matrix is symmetric and real)

        #Error in Pseudocode: Square Root + multiplication order 
        G1= ev.dot( np.diag(np.sqrt(1/eigs)) )
        G2= HXp.T.dot(G1)

        U,s,Vh=np.linalg.svd(G2)
        
        #Compute  sqrt of matrix,
        rad=(np.ones(self.Ne)-np.square(s)).astype(complex)
        rad=np.sqrt(rad)
        A=np.diag(rad)

        W1p= U.dot(A)
        W2p= W1p.dot(U.T)
        d= np.subtract(self.Y,Ymean)

        w1= ev.T.dot(d)
        w2= np.diag(1/eigs).T.dot(w1)
        w3= ev.dot(w2)
        w4= HXp.T.dot(w3)
        
        W=W2p+w4[:,None]

        return W

    def _update(self,Xmean,Xfp,W):
        """
        Update Step
        """
        Xa= np.matmul(Xmean[:,None]+Xfp,W)
        return Xa



