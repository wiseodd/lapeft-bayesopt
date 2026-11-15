import pandas as pd
import pytest
import torch

from lapeft_bayesopt.foundation_models.utils import (
    average_llm_features,
    extract_last_llm_features,
)
from lapeft_bayesopt.utils import acqf, helpers
from lapeft_bayesopt.utils.configs import LaplaceConfig
from tests.conftest import TinyT5Regressor, ToyDataProcessor


@pytest.mark.parametrize(
    "kwargs",
    [
        {"marglik_mode": "foo"},
        {"prior_prec_structure": "foo"},
        {"noise_var": -1.0},
        {"hess_factorization": "kron", "prior_prec_structure": "diagonal"},
        {"subset_of_weights": "last_layer", "prior_prec_structure": "layerwise"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LaplaceConfig(**kwargs)


def test_default_last_layer_name_is_output_layer(tokenizer, make_get_model):
    # The default must point to the head's output layer, not a hidden layer
    model = make_get_model(n_outputs=3)()
    last_layer = model.get_submodule(LaplaceConfig().last_layer_name)
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 3


def test_dataloader(tokenizer, toy_df):
    data_processor = ToyDataProcessor(tokenizer, ["y1", "y2"])
    batch = next(iter(data_processor.get_dataloader(toy_df, batch_size=4)))

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].shape == (4, 2)
    # Every prompt ends with exactly one <eos>
    assert (batch["input_ids"] == tokenizer.eos_token_id).sum(1).eq(1).all()


def test_dataloader_single_eos(eos_adding_tokenizer, toy_df):
    # The tokenizer's own <eos> and the appended one must not both be kept
    data_processor = ToyDataProcessor(eos_adding_tokenizer, "y1")
    batch = next(iter(data_processor.get_dataloader(toy_df, batch_size=4)))
    eos_id = eos_adding_tokenizer.eos_token_id
    assert (batch["input_ids"] == eos_id).sum(1).eq(1).all()


def test_regressor_forward(tokenizer, toy_df):
    data_processor = ToyDataProcessor(tokenizer, "y1")
    batch = next(iter(data_processor.get_dataloader(toy_df, batch_size=4)))
    model = TinyT5Regressor(tokenizer, n_outputs=2)
    assert model(batch).shape == (4, 2)


def test_average_llm_features():
    feats = torch.arange(12.0).reshape(1, 4, 3)
    input_ids = torch.tensor([[5, 6, 1, 0]])  # last token is padding
    out = average_llm_features(feats, input_ids, pad_token_id=0)
    assert out.shape == (1, 1, 3)
    assert torch.allclose(out, feats[:, :3].mean(1, keepdim=True))


def test_extract_last_llm_features():
    feats = torch.randn(2, 4, 3)
    input_ids = torch.tensor([[5, 6, 1, 0], [5, 6, 7, 1]])
    out = extract_last_llm_features(feats, input_ids, eos_token_id=1)
    assert torch.allclose(out, torch.stack([feats[0, 2], feats[1, 3]]))


def test_acqfs():
    f_mean, f_var = torch.randn(7), torch.rand(7)
    assert acqf.thompson_sampling(f_mean, f_var).shape == (7,)
    assert torch.allclose(
        acqf.thompson_sampling(f_mean, f_var, random_state=1),
        acqf.thompson_sampling(f_mean, f_var, random_state=1),
    )
    assert torch.allclose(acqf.ucb(f_mean, f_var, gamma=0.0), f_mean)
    assert (acqf.ei(f_mean, f_var, curr_best_val=0.0) >= 0).all()

    f_mean, f_cov = torch.randn(7, 2), torch.eye(2).expand(7, 2, 2)
    assert acqf.thompson_sampling_multivariate(f_mean, f_cov).shape == (7, 2)

    y = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    assert torch.allclose(acqf.scalarize(y), torch.tensor([2.0, 3.0]))


def test_pop_df():
    df = pd.DataFrame({"a": [1, 2, 3]})
    row = helpers.pop_df(df, 1)
    assert row["a"] == 2
    assert df["a"].tolist() == [1, 3]
    assert df.index.tolist() == [0, 1]
