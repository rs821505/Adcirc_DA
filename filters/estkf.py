import numpy as np
from k_filters import BaseFilter


class estkf(base_filter):
    """Error-subspace transform Kalman Filter

    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9

    Parameters
    ----------
    base_filter : object
        parent class to all ensemble type filters in the directory
    """

    def _assimilate(self):
        """Main run method that calls the typical forecast and anlysis steps
            of the ensemble kalman filter algorithm

        Returns
        -------
        np.ndarray
            state (analysis/posterior) vector
        """

        self._means()
        wa = self._forecast()
        state_analysis = self._analysis(wa)

        return state_analysis

    def _projection_matrix(self):
        """Create projection matrix:
        create matrix of shape Ne x Ne-1 filled with off diagonal values
        fill diagonal with diagonal values then replace values of last row

        Returns
        -------
        np.ndarray
            projection matrix
        """

        inv_ne = -1 / np.sqrt(self.ne)
        off_diag = -1 / (self.ne * (-inv_ne + 1))
        diag = 1 + off_diag

        projection_matrix = np.ones((self.ne, self.ne - 1)) * off_diag
        np.fill_diagonal(projection_matrix, diag)
        projection_matrix[-1, :] = inv_ne

        return projection_matrix

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        np.ndarray
            weight vector
        """

        projection_matrix = self._projection_matrix()  # get projection matrix
        residual = self.observations - self.model_observations.mean(
            axis=1
        )  # innovation/residual vector

        # error in pseudocode, replace L by projection_matrix
        hl = self.model_observations.dot(projection_matrix)
        b1 = np.diag(1 / self.obs_covariance).dot(hl)
        c1 = (self.ne - 1) * np.identity(self.ne - 1)
        c2 = self.inf_fact * c1 + hl.t.dot(b1)

        # EVD of c2, assumed symmetric
        eigs, u = np.linalg.eigh(c2)

        d1 = b1.t.dot(residual)
        d2 = u.t.dot(d1)
        d3 = d2 / eigs
        t = u.dot(np.diag(1 / np.sqrt(eigs)).dot(u.t))  # symmetric square root

        wm = u.dot(d3)  # mean weight
        wp = t.dot(projection_matrix.t * np.sqrt((self.ne - 1)))  # perturbation weight
        w = wm[:, None] + wp  # total weight matrix + projection matrix transform
        wa = projection_matrix.dot(w)

        return wa

    def _analysis(self, wa):
        """Analysis/posterior or best guess in the ensemble kalman filter algorithm

        Parameters
        ----------
        wa : np.ndarray
            weight vector

        Returns
        -------
        np.ndarray
            state analysis/aposterior vector
        """
        return self.state_mean[:, None] + self.centered_state_forecasts.dot(wa)
