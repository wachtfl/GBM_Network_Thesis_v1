import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader.h5_dataloader import H5GraphDataLoader
from models.model_factory import ModelFactory
from plotting.plotter import Plotter
from models.cross_validator import CrossValidator


class PipelineRunner:
    """
    Streaming, memory-efficient implementation.
    - No full concatenation of X, y, coords.
    - Two-pass scaling (partial_fit + transform).
    - Incremental CSV writing for node-level predictions.
    """

    def __init__(self, data_dir, model_name="rf", cv_method="lopo", output_dir="results", n_splits=3):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = H5GraphDataLoader(data_dir)
        self.model = ModelFactory.get(model_name)
        self.scaler = StandardScaler()
        self.crossvalidator = CrossValidator(method=cv_method, n_splits=n_splits)

    def _compute_class_weights(self, y):
        unique, counts = np.unique(y, return_counts=True)
        total = y.shape[0]
        return {int(lbl): total / (2 * cnt) for lbl, cnt in zip(unique, counts)}

    def _patient_ids(self):
        return list(self.loader.mapping.keys())

    def _load_chunk(self, pid):
        """Load one patient and return (X, y, coords)."""
        meta = self.loader.mapping[pid]["meta"]
        label = meta["label"]

        pack = self.loader.load_full_graph(pid)
        n = pack["features"].shape[0]

        X = pack["features"]
        y = np.full(n, label)
        coords = pack["coords"]
        return X, y, coords

    def _scaler_fit_pass(self):
        """First pass: compute scaling stats via partial_fit."""
        for pid in self._patient_ids():
            X, _, _ = self._load_chunk(pid)
            self.scaler.partial_fit(X)

    def run_node_level_predictor(self):

        # First pass: fit scaler
        self._scaler_fit_pass()

        # Prepare CSV writer
        node_file = self.output_dir / "node_level_predictions.csv"
        header_written = False

        patient_preds = {}
        patient_labels = {}

        # Second pass: CV loop
        for train_pids, test_pids in self.crossvalidator.split_by_patient(self._patient_ids()):

            # ----- Load train chunk-by-chunk -----
            X_train_list, y_train_list = [], []
            for pid in train_pids:
                Xp, yp, _ = self._load_chunk(pid)
                X_train_list.append(self.scaler.transform(Xp))
                y_train_list.append(yp)

            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            # ----- Class weights -----
            cw = self._compute_class_weights(y_train)
            try:
                self.model.set_params(class_weight=cw)
            except Exception:
                pass

            # ----- Fit model -----
            self.model.fit(X_train, y_train)

            # ----- Test patients -----
            for pid in test_pids:
                Xp, yp, coords = self._load_chunk(pid)
                Xp_scaled = self.scaler.transform(Xp)

                preds = self.model.predict_proba(Xp_scaled)[:, 1]

                # ----- Stream node-level CSV write -----
                with open(node_file, "a", newline="") as f:
                    writer = csv.writer(f)

                    if not header_written:
                        writer.writerow(["sample", "X", "Y", "Z", "pred", "label"])
                        header_written = True

                    for p, (x, y_, z), lbl in zip(preds, coords, yp):
                        writer.writerow([pid, float(x), float(y_), float(z), float(p), int(lbl)])

                # ----- Patient-level mean prediction -----
                patient_preds[pid] = float(np.mean(preds))
                patient_labels[pid] = int(yp[0])

                Plotter.plot_prediction_distribution(
                    preds,
                    pid,
                    str(self.output_dir / f"pred_dist_{pid}.png")
                )

        # Final aggregation
        node_df = pd.read_csv(node_file)

        summary_df = pd.DataFrame({
            "patient_id": list(patient_preds.keys()),
            "label": [patient_labels[p] for p in patient_preds.keys()],
            "mean_prediction": [patient_preds[p] for p in patient_preds.keys()],
        })

        summary_df["high_conf_ratio"] = [
            float(np.mean(node_df[node_df["sample"] == pid]["pred"] > 0.8))
            for pid in summary_df["patient_id"]
        ]

        summary_df["n_nodes"] = [
            int((node_df["sample"] == pid).sum())
            for pid in summary_df["patient_id"]
        ]

        summary_df.to_csv(self.output_dir / "analysis_summary.csv", index=False)

        # Patient-level metrics
        auc = roc_auc_score(summary_df["label"], summary_df["mean_prediction"])
        ap = average_precision_score(summary_df["label"], summary_df["mean_prediction"])

        print("AUC:", round(auc, 3))
        print("Average Precision:", round(ap, 3))

        return summary_df, node_df
