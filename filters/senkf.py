import numpy as np
from Base_Filter import base_filter


class senkf(base_filter):
    """Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),

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

        self.means()
        self.obs_cov_mat = self._obs_error_mat()

        obs_anomaly_estimate, residual = self._forecast()
        state_analysis = self._analysis(obs_anomaly_estimate, residual)

        return state_analysis

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        np.ndarray
            weight vector
        """

        np.random.seed(42)

        forecast_error_covariance = (
            self.centered_observations @ self.centered_observations.T / (self.ne - 1)
        )
        residual_covariance = forecast_error_covariance + self.obs_cov_mat
        perturbed_observations = (
            np.random.standard_normal((self.ny, self.ne))
            * np.sqrt(self.obs_covariance)[:, None]
        )
        residual = (
            self.observations[:, None]
            + perturbed_observations
            - self.model_observations
        )

        return residual_covariance, residual

    def _analysis(self, obs_anomaly_estimate, residual):
        """Analysis/posterior or best guess in the ensemble kalman filter algorithm

        Parameters
        ----------
        residual_covariance: np.ndarray
            observation anomaly estimate matrix; sample estimate of HP^fH^T matrix
        residual : np.ndarray
            residual vector

        Returns
        -------
        np.ndarray
            state analysis/aposterior vector
        """

        gain_residual = (
            self.centered_observations.T
            @ np.linalg.solve(obs_anomaly_estimate, residual)
        ) / (self.ne - 1)

        return self.state_forecast + self.centered_state_forecasts @ gain_residual
