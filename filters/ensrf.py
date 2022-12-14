import numpy as np
import Filter


class Ensrf(Filter):
    """Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 9, see section 5.6. Pseudocode has some errors, eg. in step 7 it should be sqrt(Lambda).

    Parameters
    ----------
    Filter : object
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
        self.get_shapes()
        self.obs_covariance = 0.1 * np.ones(self.ny)
        self.get_means()
        self.obs_cov_mat = self.observation_cov_matrix()

        w = self._forecast()
        state_analysis = self._analysis(w)

        return state_analysis

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        np.ndarray
            weight vector
        """

        i1 = np.matmul(
            self.centered_observations, self.centered_observations.T
        )  # Gram matrix of perturbations
        i2 = i1 + (self.ne - 1) * self.obs_cov_mat
        eigs, ev = np.linalg.eigh(
            i2
        )  # compute eigenvalues and eigenvectors (use that matrix is symmetric and real)

        # Error in Pseudocode: Square Root + multiplication order
        g1 = ev.dot(np.diag(np.sqrt(1 / eigs)))
        g2 = self.centered_observations.T.dot(g1)

        u, s, _ = np.linalg.svd(g2)

        # Compute sqrt of matrix,
        rad = np.sqrt((np.ones(self.ne) - np.square(s)).astype(complex))
        a = np.diag(rad)

        w1p = u.dot(a)
        w2p = w1p.dot(u.T)
        residual = np.subtract(self.observations, self.model_observation_mean)

        w1 = ev.T.dot(residual)
        w2 = np.diag(1 / eigs).T.dot(w1)
        w3 = ev.dot(w2)
        w4 = self.centered_observations.T.dot(w3)
        w = w2p + w4[:, None]

        return w

    def _analysis(self, w):
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
        return np.matmul(self.state_mean[:, None] + self.centered_state_forecasts, w)
