import torch
import numpy as np
import time
from Base_Filter_Torch import base_filter_torch


class senkf_torch(base_filter_torch):
    """
    Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    """

    def _assimilate(self):
        """
        Assimilate data
        obtain model dimensions, state and obseervation statistics,
        and run forecasting and analysis steps

        returns: state_analysis: ensemble analysis
        """
        self.get_shapes()
        self.r = 0.5 * torch.ones(self.ny)
        self.obs_cov_mat = self._obs_error_mat()
        self.get_means()

        a, residual = self._forecast(self.centered_observations, self.obs_cov_mat)
        state_analysis = self._analysis(
            a, residual, self.centered_state_forecasts, self.centered_observations
        )

        return state_analysis.numpy()

    def _forecast(self):
        """
        Forecast Step
        """
        torch.manual_seed(42)

        hph = self.centered_observations.matmul(self.centered_observations.T) / (
            self.ne - 1
        )

        a = hph.add(self.obs_cov_mat)

        y_p = torch.randn((self.ny, self.ne), dtype=torch.float64) * torch.sqrt(
            self.obs_covariance
        ).unsqueeze_(1)

        residual = self.observations.unsqueeze_(1).add(y_p.sub(self.model_observations))

        return a, residual

    def _analysis(self, a, residual):
        """
        Analysis Step
        returns: state_analysis: ensemble analysis
        """

        e = self.centered_observations.t().matmul(torch.linalg.solve(a, residual))

        return self.state_forecast.add(
            self.centered_state_forecasts.matmul((e / (self.ne - 1)))
        )
