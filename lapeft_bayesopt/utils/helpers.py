import pandas as pd


def pop_df(df: pd.DataFrame, idx: int):
    row = df.loc[idx]
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return row


def y_transform(y, maximize: bool):
    """
    Negating y if maximize=False.
    Rationale: we define the BayesOpt as a maximization problem.
    """
    return y if maximize else -y
