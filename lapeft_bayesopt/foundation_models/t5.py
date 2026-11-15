from __future__ import annotations

from collections.abc import Mapping

import torch
from transformers import T5Config, T5EncoderModel

from lapeft_bayesopt.foundation_models.base import BaseLLMRegressor
from lapeft_bayesopt.foundation_models.utils import LLMFeatureType


class T5Regressor(BaseLLMRegressor):
    def __init__(
        self,
        kind: str,
        tokenizer,
        reduction: LLMFeatureType = LLMFeatureType.AVERAGE,
        n_hidden_units: int = 100,
        n_outputs: int = 1,
    ):
        assert kind in [
            "t5-small",
            "t5-base",
            "t5-large",
            "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        ]

        config = T5Config.from_pretrained(kind)
        config.dropout_rate = 0
        feature_extractor = T5EncoderModel.from_pretrained(kind, config=config)

        super().__init__(
            tokenizer=tokenizer,
            reduction=reduction,
            feature_dim=feature_extractor.config.d_model,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs,
        )

        self.config = config
        self.feature_extractor = feature_extractor

    def forward_features(self, data: Mapping) -> torch.Tensor:
        input_ids = data["input_ids"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids).last_hidden_state
        return feat
