from laplace.curvature import AsdlGGN, CurvatureInterface
from typing import *


class LaplaceConfig:
    def __init__(
        self,
        batch_size: int = 20,
        lr: float = 1e-3,
        lr_lora: float = 3e-4,
        wd: float = 0.01,
        grad_clip: float = 0.0,
        n_epochs: int = 50,
        head_n_epochs: int = 100,
        marglik_mode: str = "posthoc",
        noise_var: Union[float, None] = None,
        subset_of_weights: str = "all",
        hess_factorization: str = "diag",
        prior_prec_structure: str = "layerwise",
        posthoc_marglik_iters: int = 200,
        online_marglik_freq: int = 5,
        hessian_backend: CurvatureInterface = AsdlGGN,
        last_layer_name: str = "base_model.model.head.modules_to_save.default.2",
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.lr_lora = lr_lora
        self.wd = wd
        self.grad_clip = grad_clip
        self.n_epochs = n_epochs
        self.head_n_epochs = head_n_epochs
        self.marglik_mode = marglik_mode
        self.noise_var = noise_var
        self.subset_of_weights = subset_of_weights
        self.hess_factorization = hess_factorization
        assert prior_prec_structure in ["scalar", "layerwise", "diagonal"]
        self.prior_prec_structure = prior_prec_structure
        self.posthoc_marglik_iters = posthoc_marglik_iters
        self.online_marglik_freq = online_marglik_freq
        self.hessian_backend = hessian_backend
        self.last_layer_name = last_layer_name
