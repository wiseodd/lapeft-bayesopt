from __future__ import annotations

import pandas as pd
import torch.utils.data as data_utils
from datasets import Dataset  # huggingface datasets
from datasets.utils.logging import disable_progress_bar
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from lapeft_bayesopt.problems.prompting import PromptBuilder

disable_progress_bar()


class DataProcessor:
    """
    Base class for all Bayesian optimization datasets (always regression).
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        num_outputs: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.prompt_builder = prompt_builder
        self.num_outputs = num_outputs
        self.tokenizer = tokenizer

        # To be defined in subclasses
        self.x_col: str | None = None
        self.target_col: str | list[str] | None = None
        self.obj_str: str | None = None
        self.maximization: bool | list[bool] | None = None

    def get_dataloader(
        self,
        pandas_dataset: pd.DataFrame,
        batch_size: int = 16,
        max_seq_len: int = 512,
        shuffle: bool = False,
        append_eos: bool = True,
    ) -> data_utils.DataLoader:
        dataset = Dataset.from_pandas(pandas_dataset)

        def tokenize(row):
            prompt = self.prompt_builder.get_prompt(row[self.x_col], self.obj_str)
            if append_eos:
                prompt += self.tokenizer.eos_token
            out = dict(self.tokenizer(prompt, truncation=True, max_length=max_seq_len))
            # Some tokenizers (e.g. T5 in transformers v5) add <eos> on their own,
            # even after the one we appended. Keep only one.
            ids = out["input_ids"]
            eos_id = self.tokenizer.eos_token_id
            if append_eos and len(ids) >= 2 and ids[-1] == ids[-2] == eos_id:
                out = {k: v[:-1] for k, v in out.items()}
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

    def _get_targets(self, row: pd.Series | dict) -> list[float]:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        targets: list of floats
            Regression target(s). Length self.num_outputs.
        """
        if isinstance(self.target_col, list):
            return [row[col] for col in self.target_col]
        else:
            return [row[self.target_col]]

    def _get_columns_to_remove(self) -> list[str]:
        """
        Returns:
        --------
        cols: list of strs
            Columns to remove from the dataset
        """
        raise NotImplementedError
