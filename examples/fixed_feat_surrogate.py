from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import torch
from torch import nn, optim
from torch import distributions as dists
import torch.utils.data as data_utils
import botorch.models.model as botorch_model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from laplace import Laplace
from laplace.curvature import BackPackGGN
from laplace.marglik_training import marglik_training
from typing import *
import math


class LaplaceBoTorch(botorch_model.Model):
    """
    BoTorch surrogate model with a Laplace-approximated Bayesian
    neural network. The Laplace class is defined in the library
    laplace-torch; install via:
    `pip install https://github.com/aleximmer/laplace.git`.

    Args:
    -----
    get_net: function None -> nn.Module
        Function that doesn't take any args and return a PyTorch model.
        Prefer torch.nn.Sequential model due to BackPACK dependency.
        Example usage: `get_net=lambda: nn.Sequential(...)`.

    train_X : torch.Tensor
        Training inputs of size (n_data, ...).

    train_Y : torch.Tensor
        Training targets of size (n_data, n_tasks).

    bnn : Laplace, optional, default=None
        When creating a new model from scratch, leave this at None.
        Use this only to update this model with a new observation during BO.

    likelihood : {'regression', 'classification'}
        Indicates whether the problem is regression or classification.

    noise_var : float | None, default=None.
        Output noise variance. If float, must be >= 0. If None,
        it is learned by marginal likelihood automatically.

    last_layer : bool, default False
        Whether to do last-layer Laplace. If True, then the model used is the
        so-called "neural linear" model.

    hess_factorization : {'full', 'diag', 'kron'}, default='kron'
        Which Hessian factorization to use to do Laplace. 'kron' provides the best
        tradeoff between speed and approximation error.

    marglik_mode : {'posthoc', 'online'}, default='posthoc'
        Whether to do online marginal-likelihood training or do standard NN training
        and use marglik to optimize hyperparams post-hoc.

    posthoc_marglik_iters: int > 0, default=100
        Number of iterations of post-hoc marglik tuning.

    online_marglik_freq: int > 0 default=50
        How often (in terms of training epoch) to do online marglik tuning.

    batch_size : int, default=64
        Batch size to use for the NN training and doing Laplace.

    n_epochs : int, default=500
        Number of epochs for training the NN.

    lr : float, default=1e-1
        Learning rate to use for training the NN.

    wd : float, default=1e-3
        Weight decay for training the NN.

    device : {'cpu', 'cuda'}, default='cpu'
        Which device to run the experiment on.
    """

    def __init__(
        self,
        get_net: Callable[[], nn.Module],
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        bnn: Laplace = None,
        likelihood: str = "regression",
        noise_var: float | None = None,
        last_layer: bool = False,
        hess_factorization: str = "kron",
        marglik_mode: str = "posthoc",
        posthoc_marglik_iters: int = 100,
        online_marglik_freq: int = 50,
        batch_size: int = 20,
        n_epochs: int = 500,
        lr: float = 1e-3,
        wd: float = 5e-4,
        device: str = "cpu",
    ):
        super().__init__()

        self.train_X = train_X
        self.train_Y = train_Y
        assert likelihood in ["regression"]  # For now
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.last_layer = last_layer
        self.subset_of_weights = "last_layer" if last_layer else "all"
        self.hess_factorization = hess_factorization
        self.posthoc_marglik_iters = posthoc_marglik_iters
        self.online_marglik_freq = online_marglik_freq
        assert device in ["cpu", "cuda"]
        self.device = device
        assert marglik_mode in ["posthoc", "online"]
        self.marglik_mode = marglik_mode
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd
        self.get_net = get_net
        self.bnn = bnn

        if type(noise_var) != float and noise_var is not None:
            raise ValueError("Noise variance must be float >= 0. or None")
        if type(noise_var) == float and noise_var < 0:
            raise ValueError("Noise variance must be >= 0.")
        self.noise_var = noise_var

        # Initialize Laplace
        if self.bnn is None:
            self._train_model(self._get_train_loader())

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        mean_y, var_y = self.get_prediction(X, use_test_loader=False, joint=False)
        if len(var_y.shape) != 3:  # Single objective
            return dists.Normal(mean_y, var_y.squeeze(-1))
        else:  # Multi objective
            return dists.MultivariateNormal(mean_y, var_y)

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs: Any
    ) -> LaplaceBoTorch:
        # Append new observation to the current data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y], dim=0)

        # Update Laplace with the updated data
        train_loader = self._get_train_loader()
        self._train_model(train_loader)

        return LaplaceBoTorch(
            # Replace the dataset & retrained BNN
            get_net=self.get_net,
            train_X=self.train_X,  # Important!
            train_Y=self.train_Y,  # Important!
            bnn=self.bnn,  # Important!
            likelihood=self.likelihood,
            noise_var=self.noise_var,
            last_layer=self.last_layer,
            hess_factorization=self.hess_factorization,
            marglik_mode=self.marglik_mode,
            posthoc_marglik_iters=self.posthoc_marglik_iters,
            online_marglik_freq=self.online_marglik_freq,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            wd=self.wd,
            device=self.device,
        )

    def get_prediction(self, test_X: torch.Tensor, joint=True, use_test_loader=False):
        """
        Batched Laplace prediction.

        Args:
        -----
        test_X: torch.Tensor
            Array of size `(batch_shape, feature_dim)`.

        joint: bool, default=True
            Whether to do joint predictions (like in GP).

        use_test_loader: bool, default=False
            Set to True if your test_X is large.


        Returns:
        --------
        mean_y: torch.Tensor
            Tensor of size `(batch_shape, num_tasks)`.

        cov_y: torch.Tensor
            Tensor of size `(batch_shape*num_tasks, batch_shape*num_tasks)`
            if joint is True. Otherwise, `(batch_shape, num_tasks, num_tasks)`.
        """
        if self.bnn is None:
            raise Exception("Train your model first before making prediction!")

        if not use_test_loader:
            mean_y, cov_y = self.bnn(test_X.to(self.device), joint=joint)
        else:
            test_loader = data_utils.DataLoader(
                data_utils.TensorDataset(test_X, torch.zeros_like(test_X)),
                batch_size=256,
            )

            mean_y, cov_y = [], []

            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                _mean_y, _cov_y = self.bnn(X_batch, joint=joint)
                mean_y.append(_mean_y)
                cov_y.append(_cov_y)

            mean_y = torch.cat(mean_y, dim=0).squeeze()
            cov_y = torch.cat(cov_y, dim=0).squeeze()

        return mean_y, cov_y

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return self.train_Y.shape[-1]

    def _train_model(self, train_loader):
        del self.bnn

        if self.marglik_mode == "posthoc":
            self._posthoc_laplace(train_loader)
        else:
            # Online
            la, model, _, _ = marglik_training(
                # Ensure that the base net is re-initialized
                self.get_net(),
                train_loader,
                likelihood=self.likelihood,
                hessian_structure=self.hess_factorization,
                n_epochs=self.n_epochs,
                backend=BackPackGGN,
                optimizer_kwargs={"lr": self.lr},
                scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
                scheduler_kwargs={"T_max": self.n_epochs * len(train_loader)},
                marglik_frequency=self.online_marglik_freq,
                # enable_backprop=True  # Important!
            )
            self.bnn = la

        # Override sigma_noise if self.noise_var is not None
        if self.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(self.noise_var)

    def _posthoc_laplace(self, train_loader):
        net = self.get_net()  # Ensure that the base net is re-initialized
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.n_epochs * len(train_loader)
        )
        loss_func = (
            nn.MSELoss() if self.likelihood == "regression" else nn.CrossEntropyLoss()
        )

        for _ in range(self.n_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = net(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        net.eval()
        self.bnn = Laplace(
            net,
            self.likelihood,
            subset_of_weights=self.subset_of_weights,
            hessian_structure=self.hess_factorization,
            backend=BackPackGGN,
        )
        self.bnn.fit(train_loader)

        if self.likelihood == "classification":
            self.bnn.optimize_prior_precision(n_steps=self.posthoc_marglik_iters)
        else:
            # For regression, tune prior precision and observation noise
            log_prior, log_sigma = (
                torch.ones(1, requires_grad=True),
                torch.ones(1, requires_grad=True),
            )
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
            for _ in range(self.posthoc_marglik_iters):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.bnn.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

    def _get_train_loader(self):
        return data_utils.DataLoader(
            data_utils.TensorDataset(self.train_X, self.train_Y),
            batch_size=self.batch_size,
            shuffle=True,
        )
