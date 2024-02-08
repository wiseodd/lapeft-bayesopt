from __future__ import annotations
import torch
import pandas as pd
from laplace import Laplace

from abc import ABC
from typing import Callable, Tuple, List

from lapeft_bayesopt.utils.configs import LaplaceConfig
from lapeft_bayesopt.problems.data_processor import DataProcessor


class LAPEFTBayesOpt(ABC):
    """
    Base class for LAPEFT BayesOpt variants.

    Parameters:
    -----------
    get_net: function of None -> torch.nn.Module
        A function returning the freshly-initialized regression NN
        attached on top of the LLM feature extractor.

    llm_feature_extractor: torch.nn.Module
        The LLM feature extractor. It takes strings and outputs feature vectors.

    training_set: list of pd.Series
        Initial training data. E.g. obtained via random search.

    data_processor: DataProcessor
        Data processor to process the pandas training set.

    bnn: Laplace, optional, default=None
        When creating a new model from scratch, leave this at None.
        Use this only to update this model with a new observation during BayesOpt run.

    laplace_config: LaplaceConfig, optional, default=None
        Override configs for Laplace
    """
    def __init__(
        self,
        get_model: Callable[[], torch.nn.Module],
        training_set: List[pd.Series],
        data_processor: DataProcessor,
        bnn: Laplace = None,
        laplace_config: LaplaceConfig = None,
        device: str = 'cuda',
    ) -> None:
        self.get_model = get_model
        self.training_set = training_set
        self.data_processor = data_processor
        self.bnn = bnn
        self.laplace_config = laplace_config if laplace_config is not None else LaplaceConfig()
        self.device = device

        if self.bnn is None:
            self.train_model()

    def train_model(self) -> None:
        """
        Train the netwok from `self.get_net` and possibly also the LLM using `self.training_set`.
        """
        raise NotImplementedError

    def posterior(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given input tensors X, obtain the Laplace posterior predictive distribution
        p(g(X) | X_train, y_train), which is a Gaussian.

        Parameters:
        -----------
        input_ids: torch.Tensor
            Shape (batch_size, seq_len)

        attention_mask: torch.Tensor
            Shape (batch_size, seq_len)

        Returns:
        --------
        posterior: torch.distributions.Normal
            Where the mean and variance are (batch_size, n_tasks)
        """
        raise NotImplementedError

    def condition_on_observations(
        self, obs: pd.DataFrame
    ) -> LAPEFTBayesOpt:
        NotImplementedError

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return self.data_processor.num_outputs
