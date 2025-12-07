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

    def split_by_patient(self, patient_ids):
        """
        patient_ids: list/array where each entry is the patient ID of sample i.
        Returns train_pids, test_pids (each as sets of patient IDs).
        """
        # Unique patient list in a stable order
        patients = list(dict.fromkeys(patient_ids))

        if self.method == "lopo":
            for test_pid in patients:
                train_pids = [p for p in patients if p != test_pid]
                yield train_pids, [test_pid]

        elif self.method == "kfold":
            # KFold operates at patient level
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=42
            )
            for train_idx, test_idx in kf.split(patients):
                train_pids = [patients[i] for i in train_idx]
                test_pids = [patients[i] for i in test_idx]
                yield train_pids, test_pids

        else:
            raise ValueError(f"Unsupported validation method: {self.method}")