from __future__ import annotations

from typing import Any

import pandas as pd


def pop_df(df: pd.DataFrame, idx: int) -> pd.Series:
    row = df.loc[idx]
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return row


def y_transform(y: Any, maximize: bool) -> Any:
    """
    Negating y if maximize=False.
    Rationale: we define the BayesOpt as a maximization problem.
    """
    return y if maximize else -y
