import torch
import numpy as np
from .Base_Filter_Torch import base_filter_torch


class estkf_torch(base_filter_torch):
    """Error-subspace transform Kalman Filter

    Pytorch implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9

    Parameters
    ----------
    base_filter_torch : object
        parent class to all ensemble type filters in the directory
    """

    def _assimilate(self):
        """Main run method that calls the typical forecast and anlysis steps
            of the ensemble kalman filter algorithm

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            state (analysis/posterior) vector, posterior covariance
        """

        self.parameters_torch(
            self.state_forecast, self.model_observations, self.observations
        )

        self.get_shapes()
        self.obs_covariance = 0.1 * torch.ones(self.ny)
        self.get_means()

        wa, cv = self._forecast()
        state_analysis = self._analysis(wa)

        return state_analysis.numpy(), cv.numpy()

    def _projection_matrix(self):
        """Create projection matrix:
        create matrix of shape ne x ne-1 filled with off diagonal values
        fill diagonal with diagonal values then replace values of last row

        Returns
        -------
        torch.Tensor
            projection Tensor
        """

        inv_ne = -1 / torch.sqrt(self.ne)
        off_diag = -1 / (self.ne * (-inv_ne + 1))
        diag = 1 + off_diag

        projection_matrix = (
            torch.ones((self.ne, self.ne - 1), dtype=torch.float32) * off_diag
        )
        projection_matrix.fill_diagonal_(diag)
        projection_matrix[-1, :] = inv_ne
        return projection_matrix

    def _forecast(self):
        """Forecast/apriori or first guess step in the ensemble kalman filter algorithm

        Returns
        -------
        torch.Tensor
            weight vector
        """

        projection_matrix = self._projection_matrix()  # get projection matrix
        residual = torch.sub(
            self.observations, self.model_observations.mean(axis=1)
        )  # innovation vector

        hl = self.model_observations.matmul(projection_matrix)
        b1 = torch.diag(1 / self.obs_covariance).matmul(hl)
        c1 = (self.ne - 1) * torch.eye(self.ne - 1)
        c2 = self.inf_fact * c1 + hl.t().matmul(b1)

        eigs, u = torch.linalg.eigh(c2)  # evd of c2, assumed symmetric

        d1 = b1.t().matmul(residual)
        d2 = u.t().matmul(d1)
        d3 = d2 / eigs
        t = u.matmul(
            torch.diag(1 / torch.sqrt(eigs)).matmul(u.t())
        )  # symmetric square root matrix

        wm = u.matmul(d3)  # mean weight
        wp = t.matmul(
            projection_matrix.t() * torch.sqrt((self.ne - 1))
        )  # perturbation weight
        w = wm.unsqueeze_(1).add(
            wp
        )  # total weight matrix + projection matrix transform
        wa = projection_matrix.matmul(w)

        return wa, c2

    def _analysis(self, wa):
        """Analysis/posterior or best guess in the ensemble kalman filter algorithm

        Parameters
        ----------
        wa : torch.Tensor
            weight vector

        Returns
        -------
        torch.Tensor
            state analysis/aposterior vector
        """
        return self.state_mean.add(self.centered_state_forecasts.matmul(wa))
