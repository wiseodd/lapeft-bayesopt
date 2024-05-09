import torch
import torch.utils.data as data_utils
import pandas as pd
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer
from datasets import Dataset  # huggingface datasets
from typing import *

from .prompting import PromptBuilder

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class DataProcessor:
    """
    Base class for all Bayesian optimization datasets (always regression).
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        num_outputs: int,
        tokenizer: PreTrainedTokenizer,
    ):
        self.prompt_builder = prompt_builder
        self.num_outputs = num_outputs
        self.tokenizer = tokenizer

        # To be defined in subclasses
        self.x_col = None
        self.target_col = None
        self.obj_str = None
        self.maximization = None

    def get_dataloader(
        self,
        pandas_dataset: pd.DataFrame,
        batch_size=16,
        max_seq_len=512,
        shuffle=False,
        append_eos=True,
    ) -> data_utils.DataLoader:
        dataset = Dataset.from_pandas(pandas_dataset)

        def tokenize(row):
            prompt = self.prompt_builder.get_prompt(row[self.x_col], self.obj_str)
            if append_eos:
                prompt += self.tokenizer.eos_token
            out = self.tokenizer(prompt, truncation=True, max_length=max_seq_len)
            out["labels"] = self._get_targets(row)
            return out

        dataset = dataset.map(
            tokenize, remove_columns=self._get_columns_to_remove(), num_proc=4
        )

        return data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def _get_targets(self, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        targets: torch.Tensor
            Regression target(s). Shape (self.num_outputs,).
        """
        if isinstance(self.target_col, list):
            return [row[col] for col in self.target_col]
        else:
            return [row[self.target_col]]

    def _get_columns_to_remove(self) -> List[str]:
        """
        Returns:
        --------
        cols: list of strs
            Columns to remove from the dataset
        """
        raise NotImplementedError
