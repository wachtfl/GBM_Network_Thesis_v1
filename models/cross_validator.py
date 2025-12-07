import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold

class CrossValidator:
    def __init__(self, method="lopo", n_splits=3):
        self.method = method
        self.n_splits = n_splits
        print(f"[CV INIT] Using method '{self.method}' with n_splits={self.n_splits}")

    def split(self, X, y, groups=None):
        if self.method == "lopo":
            logo = LeaveOneGroupOut()
            print("[CV SPLIT] Using Leave-One-Group-Out")
            return logo.split(X, y, groups)
        elif self.method == "kfold":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            print("[CV SPLIT] Using KFold")
            return kf.split(X)
        else:
            raise ValueError("Unsupported validation method")

    def split_by_patient(self, patient_ids):
        """
        patient_ids: list/array where each entry is the patient ID of sample i.
        Returns train_pids, test_pids (each as sets of patient IDs).
        """
        patients = list(dict.fromkeys(patient_ids))
        print(f"[CV SPLIT BY PATIENT] {len(patients)} unique patients detected")

        if self.method == "lopo":
            for idx, test_pid in enumerate(patients):
                train_pids = [p for p in patients if p != test_pid]
                print(f"[CV SPLIT] Split {idx + 1}: train={train_pids}, test={[test_pid]}")
                yield train_pids, [test_pid]

        elif self.method == "kfold":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            for idx, (train_idx, test_idx) in enumerate(kf.split(patients)):
                train_pids = [patients[i] for i in train_idx]
                test_pids = [patients[i] for i in test_idx]
                print(f"[CV SPLIT] Fold {idx + 1}: train={train_pids}, test={test_pids}")
                yield train_pids, test_pids

        else:
            raise ValueError(f"Unsupported validation method: {self.method}")
