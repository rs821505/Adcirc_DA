import torch
import numpy as np
import time
from Base_Filter_Torch import base_filter_torch


class senkf_torch(base_filter_torch):
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
        torch.Tensor
            state (analysis/posterior) vector
        """
        self.parameters_torch(
            self.state_forecast, self.model_observations, self.observations
        )
        self.get_shapes()
        self.obs_covariance = 0.1 * torch.ones(self.ny)
        self.obs_cov_mat = self._obs_error_mat()
        self.get_means()

        residual_covariance, residual = self._forecast()
        state_analysis = self._analysis(residual_covariance, residual)

        return state_analysis.numpy()

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        torch.Tensor
            weight vector
        """

        torch.manual_seed(42)

        forecast_error_covariance = self.centered_observations.matmul(
            self.centered_observations.T
        ) / (self.ne - 1)

        residual_covariance = forecast_error_covariance.add(self.obs_cov_mat)

        perturbed_observations = torch.randn(
            (self.ny, self.ne), dtype=torch.float32
        ) * torch.sqrt(self.obs_covariance).unsqueeze_(1)

        residual = self.observations.unsqueeze_(1).add(
            perturbed_observations.sub(self.model_observations)
        )

        return residual_covariance, residual

    def _analysis(self, residual_covariance, residual):
        """Analysis/posterior or best guess in the ensemble kalman filter algorithm

        Parameters
        ----------
        residual_covariance: torch.Tensor
            residual_covariance matrix; sample estimate of HP^fH^T matrix
        residual : np.torch.Tensor
            residual vector

        Returns
        -------
        torch.Tensor
            state analysis/aposterior vector
        """
        e = self.centered_observations.t().matmul(
            torch.linalg.solve(residual_covariance, residual)
        )

        return self.state_forecast.add(
            self.centered_state_forecasts.matmul((e / (self.ne - 1)))
        )
