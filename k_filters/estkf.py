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
        xa = self._analysis(xbar, xfp, Wa)

        return xa

    def _projection_matrix(self):
        """
        Create projection matrix:
        create matrix of shape Ne x Ne-1 filled with off diagonal values
        fill diagonal with diagonal values then replace values of last row
        """

        inv_ne = -1 / np.sqrt(self.ne)
        off_diag = -1 / (self.ne * (-inv_ne + 1))
        diag = 1 + off_diag

        a = np.ones((self.ne, self.ne - 1)) * off_diag
        np.fill_diagonal(a, diag)
        a[-1, :] = inv_ne

        return a

    def _forecast(self):
        """
        Forecast Step
        """

        a = self._projection_matrix()  # get projection matrix

        d = self.y - self.model_observations.mean(axis=1)  # innovation vector

        # error in pseudocode, replace L by a
        hl = self.model_observations.dot(a)
        b1 = np.diag(1 / self.obs_covariance).dot(hl)
        c1 = (self.ne - 1) * np.identity(self.ne - 1)
        c2 = self.inf_fact * c1 + hl.T.dot(b1)

        # EVD of c2, assumed symmetric
        eigs, u = np.linalg.eigh(c2)

        d1 = b1.T.dot(d)
        d2 = u.T.dot(d1)
        d3 = d2 / eigs
        T = u.dot(np.diag(1 / np.sqrt(eigs)).dot(u.T))  # symmetric square root

        wm = u.dot(d3)  # mean weight
        Wp = T.dot(a.T * np.sqrt((self.ne - 1)))  # perturbation weight
        W = wm[:, None] + Wp  # total weight matrix + projection matrix transform
        Wa = a.dot(W)

        return Wa

    def _analysis(self, xbar, xfp, Wa):
        """
        Update Step
        """
        Xa = xbar[:, None] + xfp.dot(Wa)  # Analysis ensemble

        return Xa
