import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold

class CrossValidator:
    def __init__(self, method="lopo", n_splits=3):
        self.method = method
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        if self.method == "lopo":
            logo = LeaveOneGroupOut()
            return logo.split(X, y, groups)
        elif self.method == "kfold":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return kf.split(X)
        else:
            raise ValueError("Unsupported validation method")
