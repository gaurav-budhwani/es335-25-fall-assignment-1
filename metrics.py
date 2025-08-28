
from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    n = len(y_hat)
    if n == 0:
        return 0.0
    correct_predictions = (y_hat == y).sum()
    return (correct_predictions / n)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()
    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    n = len(y_hat)
    if n == 0:
        return 0.0
    mse = ((y_hat - y) ** 2).mean()
    return mse ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    n = len(y_hat)
    if n == 0:
        return 0.0
    mae_value = (y_hat - y).abs().mean()
    return mae_value
