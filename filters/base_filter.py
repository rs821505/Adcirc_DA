import inspect
from dataclasses import dataclass, field
from pprint import pprint
import numpy as np

# import torch


@dataclass(frozen=True, order=True)
class base_filter:

    state_forecast: np.ndarray = None
    model_observations: np.ndarray = None
    observations: np.ndarray = None
    obs_covariance: np.ndarray = None
    inf_fact: float = 1.65

    def get_shape(self):
        self.ny = np.size(self.observations)
        self.nx, self.ne = self.state_forecast.shape

    def get_means(self):

        self.state_mean = self.state_forecast.mean(axis=1)
        self.observation_mean = self.model_observations.mean(axis=1)

        self.zero_mean_state = self.state_forecast - self.state_mean[:None]
        self.zero_mean_observations = self.observations - np.expand_dims(
            self.observation_mean, axis=1
        )

    def observation_cov_matrix(self):
        return np.diag(self.obs_covariance)
