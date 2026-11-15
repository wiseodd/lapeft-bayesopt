import pytest
import torch

from lapeft_bayesopt.surrogates.lora import LAPEFTBayesOptLoRA
from lapeft_bayesopt.utils.configs import LaplaceConfig
from tests.conftest import ToyDataProcessor

N_TRAIN = 8


def _fast_config(**kwargs) -> LaplaceConfig:
    return LaplaceConfig(
        batch_size=4, n_epochs=2, head_n_epochs=2, posthoc_marglik_iters=5, **kwargs
    )


def _fit(get_model, df, data_processor, config):
    training_set = [df.iloc[i] for i in range(N_TRAIN)]
    return LAPEFTBayesOptLoRA(
        get_model, training_set, data_processor, laplace_config=config, device="cpu"
    )


@pytest.mark.parametrize(
    "marglik_mode,hess_factorization,subset_of_weights,prior_prec_structure",
    [
        ("posthoc", "kron", "all", "layerwise"),
        ("posthoc", "diag", "all", "diagonal"),
        ("posthoc", "full", "last_layer", "scalar"),
        ("posthoc", "kron", "last_layer", "scalar"),
        ("online", "kron", "all", "layerwise"),
        ("online", "diag", "all", "diagonal"),
    ],
)
@pytest.mark.parametrize("noise_var", [None, 1e-3])
def test_posterior(
    tokenizer,
    toy_df,
    make_get_model,
    marglik_mode,
    hess_factorization,
    subset_of_weights,
    prior_prec_structure,
    noise_var,
):
    data_processor = ToyDataProcessor(tokenizer, "y1")
    config = _fast_config(
        marglik_mode=marglik_mode,
        hess_factorization=hess_factorization,
        subset_of_weights=subset_of_weights,
        prior_prec_structure=prior_prec_structure,
        noise_var=noise_var,
    )
    model = _fit(make_get_model(), toy_df, data_processor, config)

    if noise_var is not None:
        assert torch.allclose(model.bnn.sigma_noise, torch.tensor(noise_var).sqrt())

    loader = data_processor.get_dataloader(toy_df.iloc[N_TRAIN:], batch_size=5)
    batch = next(iter(loader))
    posterior = model.posterior(batch)

    assert posterior.mean.shape == (5, 1)
    assert posterior.variance.shape == (5, 1)
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.variance > 0).all()


def test_posterior_variance_is_not_squared(tokenizer, toy_df, make_get_model):
    # `posterior.variance` must be the predictive variance, not its square
    data_processor = ToyDataProcessor(tokenizer, "y1")
    model = _fit(make_get_model(), toy_df, data_processor, _fast_config())

    batch = next(iter(data_processor.get_dataloader(toy_df.iloc[N_TRAIN:])))
    f_mean, f_var = model.bnn(batch)
    expected_var = f_var.squeeze(-1) + model.bnn.sigma_noise**2

    posterior = model.posterior(batch)
    assert torch.allclose(posterior.mean, f_mean)
    assert torch.allclose(posterior.variance, expected_var)


def test_posterior_multioutput(tokenizer, toy_df, make_get_model):
    data_processor = ToyDataProcessor(tokenizer, ["y1", "y2"])
    model = _fit(make_get_model(n_outputs=2), toy_df, data_processor, _fast_config())

    batch = next(iter(data_processor.get_dataloader(toy_df.iloc[N_TRAIN:])))
    _, f_cov = model.bnn(batch)
    posterior = model.posterior(batch)

    B = batch["input_ids"].shape[0]
    assert posterior.mean.shape == (B, 2)
    assert posterior.variance.shape == (B, 2)
    expected_var = torch.diagonal(f_cov, dim1=-2, dim2=-1) + model.bnn.sigma_noise**2
    assert torch.allclose(posterior.variance, expected_var)


def test_condition_on_observations(tokenizer, toy_df, make_get_model):
    data_processor = ToyDataProcessor(tokenizer, "y1")
    config = _fast_config(noise_var=1e-3)
    model = _fit(make_get_model(), toy_df, data_processor, config)

    new_model = model.condition_on_observations(toy_df.iloc[N_TRAIN])

    assert len(new_model.training_set) == N_TRAIN + 1
    assert new_model.laplace_config is config
    assert new_model.device == "cpu"
    assert new_model.dtype == "float32"
    assert new_model.bnn is not None
