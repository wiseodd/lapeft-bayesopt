from __future__ import annotations

from dataclasses import dataclass

from laplace.curvature import CurvatureInterface, CurvlinopsGGN


@dataclass
class LaplaceConfig:
    batch_size: int = 20
    lr: float = 1e-3
    lr_lora: float = 3e-4
    wd: float = 0.01
    grad_clip: float = 0.0
    n_epochs: int = 50
    head_n_epochs: int = 100
    marglik_mode: str = "posthoc"
    noise_var: float | None = None
    subset_of_weights: str = "all"
    hess_factorization: str = "diag"
    prior_prec_structure: str = "layerwise"
    posthoc_marglik_iters: int = 200
    online_marglik_freq: int = 5
    hessian_backend: type[CurvatureInterface] = CurvlinopsGGN
    last_layer_name: str = "base_model.model.head.modules_to_save.default.4"

    def __post_init__(self) -> None:
        if self.marglik_mode not in ["posthoc", "online"]:
            raise ValueError("marglik_mode must be 'posthoc' or 'online'")
        if self.prior_prec_structure not in ["scalar", "layerwise", "diagonal"]:
            raise ValueError(
                "prior_prec_structure must be 'scalar', 'layerwise', or 'diagonal'"
            )
        if (
            self.subset_of_weights == "last_layer"
            and self.prior_prec_structure == "layerwise"
        ):
            raise ValueError(
                "Last-layer Laplace supports only a 'scalar' or 'diagonal' prior_prec_structure"
            )
        if (
            self.hess_factorization == "kron"
            and self.prior_prec_structure == "diagonal"
        ):
            raise ValueError(
                "Kronecker-factored Laplace supports only a 'scalar' or 'layerwise' prior_prec_structure"
            )
        if self.noise_var is not None and self.noise_var < 0:
            raise ValueError("noise_var must be >= 0 or None")
