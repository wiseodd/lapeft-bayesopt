from transformers import T5EncoderModel, T5Config
from .base import BaseLLMRegressor
from .utils import LLMFeatureType


class T5Regressor(BaseLLMRegressor):
    def __init__(
        self,
        kind,
        tokenizer,
        reduction=LLMFeatureType.AVERAGE,
        n_hidden_units=100,
        n_outputs=1,
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

    def forward_features(self, data):
        input_ids = data["input_ids"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids).last_hidden_state
        return feat
