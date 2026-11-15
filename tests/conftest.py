import string

import numpy as np
import pandas as pd
import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

from lapeft_bayesopt.foundation_models.base import BaseLLMRegressor
from lapeft_bayesopt.problems.data_processor import DataProcessor
from lapeft_bayesopt.problems.prompting import PromptBuilder

torch.manual_seed(9999)


def _make_tokenizer(add_eos: bool = False) -> PreTrainedTokenizerFast:
    """
    Character-level tokenizer, so that the tests don't download anything.
    With `add_eos=True` it appends `</s>` to every input, like T5's tokenizer.
    """
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    vocab |= {c: i + len(vocab) for i, c in enumerate(string.printable)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    if add_eos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>", special_tokens=[("</s>", 1)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )


class TinyT5Regressor(BaseLLMRegressor):
    def __init__(self, tokenizer, n_outputs: int = 1):
        config = T5Config(
            vocab_size=len(tokenizer),
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.0,
        )
        super().__init__(
            tokenizer=tokenizer,
            feature_dim=config.d_model,
            n_hidden_units=8,
            n_outputs=n_outputs,
        )
        self.feature_extractor = T5EncoderModel(config)

    def forward_features(self, data):
        device = next(self.parameters()).device
        return self.feature_extractor(data["input_ids"].to(device)).last_hidden_state


class IdentityPromptBuilder(PromptBuilder):
    def get_prompt(self, x: str, obj_str: str) -> str:
        return x


class ToyDataProcessor(DataProcessor):
    def __init__(self, tokenizer, target_col: str | list[str]):
        num_outputs = len(target_col) if isinstance(target_col, list) else 1
        super().__init__(IdentityPromptBuilder("just-x"), num_outputs, tokenizer)
        self.x_col = "x"
        self.target_col = target_col
        self.obj_str = "y"
        self.maximization = True

    def _get_columns_to_remove(self) -> list[str]:
        return ["x", "y1", "y2"]


@pytest.fixture(scope="session")
def tokenizer():
    return _make_tokenizer()


@pytest.fixture(scope="session")
def eos_adding_tokenizer():
    return _make_tokenizer(add_eos=True)


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(1)
    n = 24
    return pd.DataFrame(
        {
            "x": ["".join(rng.choice(list("CNO()="), 6)) for _ in range(n)],
            "y1": rng.normal(size=n),
            "y2": rng.normal(size=n),
        }
    )


@pytest.fixture
def make_get_model(tokenizer):
    def make(n_outputs: int = 1):
        def get_model():
            model = TinyT5Regressor(tokenizer, n_outputs=n_outputs)
            config = LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                modules_to_save=["head"],
            )
            lora_model = get_peft_model(model, config)
            for p in lora_model.base_model.head.original_module.parameters():
                p.requires_grad = False
            return lora_model

        return get_model

    return make
