import os
import numpy as np
import pandas as pd

from data_loader.h5_dataloader import H5GraphDataLoader
from models.classic_models.model_factory import ModelFactory
from models.cross_validator import CrossValidator
from plotting.plotter import Plotter

class PipelineRunner:
    def __init__(self, data_dir, model_name="rf", cv_method="lopo", output_dir="results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.loader = H5GraphDataLoader(data_dir)
        self.model = ModelFactory.get(model_name)
        self.cv = CrossValidator(method=cv_method)

    def run_node_level_predictor(self):
        """
        Wrap your entire logic from run_complete_analysis()
        but everything comes through data loaders + model factory.
        """

        # 1. Load all patient IDs
        patient_ids = list(self.loader.mapping.keys())

        predictions = []
        labels = []

        # Leave-one-out iteration
        for test_id in patient_ids:
            print(f"Testing patient: {test_id}")

            # TRAIN
            X_train, y_train = [], []
            for pid in patient_ids:
                if pid != test_id:
                    record = self.loader.mapping[pid]["meta"]
                    train = self.loader.load_training_nodes(pid, record["label"])
                    X_train.append(train["features"])
                    y_train.append(train["label"])

            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)

            self.model.fit(X_train, y_train)

            # TEST
            test_record = self.loader.mapping[test_id]["meta"]
            test_graph = self.loader.load_full_graph(test_id)
            preds = self.model.predict_proba(test_graph["features"])[:,1]

            Plotter.plot_prediction_distribution(
                preds, test_id, os.path.join(self.output_dir, f"dist_{test_id}.png")
            )

            predictions.append(np.mean(preds))
            labels.append(test_record["label"])

        # Overall ROC
        Plotter.plot_auc_curve(
            labels, predictions, os.path.join(self.output_dir, "roc_curve.png")
        )

        return predictions, labels
