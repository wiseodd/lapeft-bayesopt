from torch import nn
from .utils import LLMFeatureType
from .utils import average_llm_features, extract_last_llm_features


class BaseLLMRegressor(nn.Module):
    """
    Abstract class for LLM regressor surrogate model.

    Arguments:
    ----------
    tokenizer:
        Huggingface tokenizer

    reduction: LLMFeatureType.{FIRST_TOKEN, LAST_TOKEN, AVERAGE}
        How to reduce the last (batch_size, seq_len, feature_dim) LLM feature over the seq_len dim

    feature_dim: int
        Feature dimensionality (see above) of the last layer LLM features

    n_hidden_units: int
        Number of hidden units of each layer of the regression head

    n_outputs: int
        Number of regression head's outputs. Should equal the number of objectives in BO.
    """

    def __init__(
        self,
        tokenizer,
        reduction=LLMFeatureType.LAST_TOKEN,
        feature_dim=None,
        n_hidden_units=100,
        n_outputs=1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.reduction = reduction
        self.feature_dim = feature_dim
        self.n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units, self.n_outputs),
        )

    def forward_features(self, data):
        """
        Given Huggingface's batch, return the last LLM features

        Arguments:
        ----------
        data: UserDict
            A dict from Huggingface dataloader. E.g. {'input_ids': ..., 'attention_mask': ...}
        """
        raise NotImplementedError

    def forward(self, data):
        # (batch_size, seq_len, feature_dim)
        feat = self.forward_features(data)

        # Aggregate the seq dimension (batch_size, seq_len, feature_dim) -> (batch_size, 1, feature_dim)
        if self.reduction == LLMFeatureType.FIRST_TOKEN:
            feat = feat[:, 0, :]
        elif self.reduction == LLMFeatureType.LAST_TOKEN:
            feat = extract_last_llm_features(
                feat, data["input_ids"], self.tokenizer.eos_token_id
            )
        elif self.reduction == LLMFeatureType.AVERAGE:
            feat = average_llm_features(
                feat, data["input_ids"], self.tokenizer.pad_token_id
            )

        # (batch_size, feature_dim)
        feat = feat.squeeze(1)

        # n_outputs
        return self.head(feat)

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
