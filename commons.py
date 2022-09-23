from typing import Tuple
import pandas as pd
import numpy as np


def split_train_val_test(df: pd.DataFrame, random_seed: float = 0, train_slice: float = 0.6, val_slice: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = df.shape[0]
    np.random.seed(random_seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(train_slice * n)
    val_size = int(val_slice * n)
    df_train = df.iloc[idx[ : train_size]].copy()
    df_val = df.iloc[idx[train_size : train_size + val_size]].copy()
    df_test = df.iloc[idx[train_size + val_size : ]].copy()

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return (df_train, df_val, df_test)

class RegressionModel: 
    def __init__(self) -> None:
        self.w = [0]

    def extract_matrix_features(self, df: pd.DataFrame) -> np.ndarray:
        pass

    def prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        X = self.extract_matrix_features(df)
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X))
        return X


    def train(self, X: np.ndarray, y: np.array, reg: float=0) -> np.array:
        pass

    def prepare_and_train(self, df: pd.DataFrame, y: np.array, reg: float=0) -> Tuple[np.array, np.ndarray]:
        X = self.prepare_X(df)
        w = self.train(X, y, reg=reg)
        return y, X

    def predict(self, df: pd.DataFrame=None, X: np.ndarray=None) -> np.array:
        if X is None and df is None:
            raise 'Error'
        if df is not None:
            X = self.prepare_X(df)
        return X.dot(self.w)

class RegressionNormalFormModel(RegressionModel):
    def train(self, X: np.ndarray, y: np.array, reg: float=0) -> np.array:
        XTX = X.T.dot(X)

        reg_eye = np.eye(XTX.shape[0]) * reg
        XTX = XTX + reg_eye

        self.w = np.linalg.solve(XTX, X.T.dot(y))
        return self.w.copy()

def msre(y_pred: np.array, y: np.array) -> float:
    error = y_pred - y
    mse = (error ** 2)
    return np.sqrt(mse.mean())


