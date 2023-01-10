import torch
import numpy as np

from Base_Filter_Torch import base_filter_torch


class ensrf_torch(base_filter_torch):
    """Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 9, see section 5.6. Pseudocode has some errors, eg. in step 7 it should be sqrt(Lambda).

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

        w = self._forecast()
        state_analysis = self._analysis(w)

        return state_analysis.numpy()

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        torch.Tensor
            weight vector
        """

        i1 = self.centered_observations.matmul(
            self.centered_observations.t()
        )  # gram matrix of perturbations
        i2 = i1 + (self.ne - 1) * self.obs_cov_mat

        eigs, ev = torch.linalg.eigh(
            i2
        )  # compute eigenvalues and eigenvectors (use that matrix is symmetric and real)

        g1 = ev.matmul(torch.diag(torch.sqrt(1 / eigs)))
        g2 = self.centered_observations.t().matmul(g1)

        u, s, _ = torch.linalg.svd(g2)

        rad = torch.ones(self.ne).sub(torch.square(s))  # Compute  sqrt of matrix,
        rad.type(torch.complex64)
        a = torch.diag(torch.sqrt(rad))

        w1p = u.matmul(a)
        w2p = w1p.matmul(u.t())
        residual = self.observations.sub(
            self.model_observation_mean.squeeze_(1)
        )  # innovation vector

        w1 = ev.t().matmul(residual)
        w2 = torch.diag(1 / eigs).t().matmul(w1)
        w3 = ev.matmul(w2)
        w4 = self.centered_observations.t().matmul(w3)
        w = w2p.add(w4.unsqueeze_(1))

        return w

    def _analysis(self, w):
        """Analysis/posterior or best guess in the ensemble kalman filter algorithm

        Parameters
        ----------
        wa : torch.Tensor
            weight vector

        Returns
        -------
        torch.Tensor

        """
        return self.state_mean.add(self.centered_state_forecasts).matmul(w)
