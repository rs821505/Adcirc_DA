import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass(order=True)
class base_filter_torch:
    """
    Base filter class representing a single data assimilation step
    Class contaings methods shared by all filters.
    """

    state_forecast: np.ndarray = None
    model_observations: np.ndarray = None
    observations: np.ndarray = None
    obs_covariance: np.ndarray = None
    inf_fact: torch.Tensor = torch.tensor(1.65)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_torch(self, nparray):
        """Converts numpy array to pytorch tensor

        Parameters
        ----------
        nparray : np.ndarray
            numpy multidimensional array

        Returns
        -------
        torch.tensor
            pytorch tensor object
        """
        return torch.as_tensor(nparray, device=self.device, dtype=torch.float32)

    def parameters_torch(self, state_forecast, model_observations, observations):
        """Converts input parameters to torch tensors

        Parameters
        ----------
        state_forecast : np.ndarray
            state forecast vector
        model_observations : np.ndarray
            model observation vector
        observations : np.ndarray
            gauge observation vector
        """
        self.state_forecast = self.to_torch(state_forecast.copy())
        self.model_observations = self.to_torch(model_observations.copy())
        self.observations = self.to_torch(observations)

    def get_shapes(self):
        """Get shapes for data assimilation

        Returns
        ---------
        int
            ny: observation vector dimension
            nx: state forecast vector dimensions
            ne: number of members in ensemble
        """

        self.ne = torch.tensor(self.state_forecast.shape[1])
        self.nx = torch.tensor(self.state_forecast.shape[0])
        self.ny = torch.tensor(self.observations.shape[0])

    def get_means(self):
        """Assigns state ensemble mean, observation mean, and the centered state and observation matrices"""

        self.state_mean = self.state_forecast.mean(axis=1)
        self.model_observation_mean = self.model_observations.mean(axis=1)

        self.centered_state_forecasts = self.state_forecast.sub(
            self.state_mean.unsqueeze_(1)
        )
        self.centered_observations = self.model_observations.sub(
            self.model_observation_mean.unsqueeze_(1)
        )

    def _obs_error_mat(self):
        """Observation covariance matrix

        Returns
        -------
        np.ndarray
            obs_covariance: observation covariance matrix
        """
        return torch.diag(self.obs_covariance)
