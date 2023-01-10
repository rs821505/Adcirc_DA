import inspect
from dataclasses import dataclass, field
from pprint import pprint
import numpy as np


@dataclass(order=True)
class base_filter:
    """Base filter class for ensemble kalman filter algorithm"""

    state_forecast: np.ndarray = None
    model_observations: np.ndarray = None
    observations: np.ndarray = None
    obs_covariance: np.ndarray = None
    inf_fact: float = 1.65

    def get_shapes(self):
        """Get shapes for data assimilation

        Returns
        ---------
        ints
            ny: observation vector dimension
            nx: state forecast vector dimensions
            ne: number of members in ensemble
        """
        self.ny = np.size(self.observations)
        self.nx, self.ne = self.state_forecast.shape

    def get_means(self):
        """Assigns state ensemble mean, observation mean, and the centered state and observation matrices"""

        self.state_mean = self.state_forecast.mean(axis=1)
        self.model_observation_mean = self.model_observations.mean(axis=1)

        self.centered_state_forecasts = self.state_forecast - self.state_mean[:, None]
        self.centered_observations = self.model_observations - np.expand_dims(
            self.model_observation_mean, axis=1
        )

    def observation_cov_matrix(self):
        """Observation covariance matrix

        Returns
        -------
        np.ndarray
            obs_covariance: observation covariance matrix
        """
        return np.diag(self.obs_covariance)
