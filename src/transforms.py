import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class Transformer:
    def __init__(self, method):
        self.method = method
        self.lmbda = None

    def fit_transform(self, y):
        if self.method == "none":
            return y

        if self.method == "log":
            return np.log1p(y)

        if self.method == "boxcox":
            y_pos = y + 1e-6
            transformed, self.lmbda = boxcox(y_pos)
            return transformed

        if self.method == "diff":
            return np.diff(y, prepend=y[0])

    def inverse(self, y, original_first=None):
        if self.method == "none":
            return y

        if self.method == "log":
            return np.expm1(y)

        if self.method == "boxcox":
            return inv_boxcox(y, self.lmbda)

        if self.method == "diff":
            return np.cumsum(y) + original_first