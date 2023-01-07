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

        state_mean = self.state_forecast.mean(axis=1)
        observation_mean = self.model_observations.mean(axis=1)

        zero_mean_state = self.state_forecast - state_mean[:None]
        zero_mean_observation = self.observations - np.expand_dims(
            observation_mean, axis=1
        )

        return zero_mean_state, zero_mean_observation, state_mean, observation_mean

    def observation_cov_matrix(self):
        return np.diag(self.obs_covariance)
