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
    Streaming, memory-efficient implementation with progress prints.
    """

    def __init__(self, data_dir, model_name="rf", cv_method="lopo", output_dir="results", n_splits=3):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INIT] Loading data from {data_dir}")
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
        print(f"[LOAD] Loading patient {pid}")
        meta = self.loader.mapping[pid]["meta"]
        label = meta["label"]

        pack = self.loader.load_full_graph(pid)
        n = pack["features"].shape[0]

        X = pack["features"]
        y = np.full(n, label)
        coords = pack["coords"]
        print(f"[LOAD] Patient {pid}: {n} nodes, label {label}")
        return X, y, coords

    def _scaler_fit_pass(self):
        print("[SCALER] Starting first pass to fit scaler")
        for pid in self._patient_ids():
            X, _, _ = self._load_chunk(pid)
            self.scaler.partial_fit(X)
        print("[SCALER] Scaler fit completed")

    def run_node_level_predictor(self):
        print("[RUN] Node-level predictor started")

        # First pass: fit scaler
        self._scaler_fit_pass()

        node_file = self.output_dir / "node_level_predictions.csv"
        header_written = False

        patient_preds = {}
        patient_labels = {}

        print("[CV] Starting cross-validation loop")
        for split_idx, (train_pids, test_pids) in enumerate(self.crossvalidator.split_by_patient(self._patient_ids())):
            print(f"[CV] Split {split_idx + 1}: train={train_pids}, test={test_pids}")

            # ----- Load train chunk-by-chunk -----
            X_train_list, y_train_list = [], []
            for pid in train_pids:
                Xp, yp, _ = self._load_chunk(pid)
                X_train_list.append(self.scaler.transform(Xp))
                y_train_list.append(yp)

            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            print(f"[TRAIN] Training data: {X_train.shape[0]} nodes")

            # ----- Class weights -----
            cw = self._compute_class_weights(y_train)
            try:
                self.model.set_params(class_weight=cw)
                print(f"[TRAIN] Class weights set: {cw}")
            except Exception:
                print("[TRAIN] Model does not accept class_weight parameter")

            # ----- Fit model -----
            self.model.fit(X_train, y_train)
            print("[TRAIN] Model fit completed")

            # ----- Test patients -----
            for pid in test_pids:
                Xp, yp, coords = self._load_chunk(pid)
                Xp_scaled = self.scaler.transform(Xp)
                preds = self.model.predict_proba(Xp_scaled)[:, 1]

                # ----- Stream node-level CSV write -----
                print(f"[CSV] Writing predictions for patient {pid}")
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

                # ----- Plot distribution -----
                try:
                    Plotter.plot_prediction_distribution(
                        preds,
                        pid,
                        str(self.output_dir / f"pred_dist_{pid}.png")
                    )
                    print(f"[PLOT] Prediction distribution saved for {pid}")
                except Exception as e:
                    print(f"[PLOT] Failed to save plot for {pid}: {e}")

        # Final aggregation
        if node_file.exists():
            print(f"[AGGREGATE] Reading node-level CSV from {node_file}")
            node_df = pd.read_csv(node_file)
        else:
            raise FileNotFoundError(f"CSV file {node_file} does not exist")

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

        summary_file = self.output_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"[AGGREGATE] Summary saved to {summary_file}")

        # Patient-level metrics
        auc = roc_auc_score(summary_df["label"], summary_df["mean_prediction"])
        ap = average_precision_score(summary_df["label"], summary_df["mean_prediction"])
        print("[METRICS] Patient-level metrics computed")
        print("AUC:", round(auc, 3))
        print("Average Precision:", round(ap, 3))

        print("[RUN] Node-level predictor finished successfully")
        return summary_df, node_df
