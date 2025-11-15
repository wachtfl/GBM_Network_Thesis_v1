import os
import h5py
import json
import numpy as np
import glob
from tqdm import tqdm

class H5GraphDataLoader:
    def __init__(self, data_dir, graph_pattern="*_v3.h5", metadata_file="metadata2.json"):
        self.data_dir = data_dir
        self.graph_pattern = os.path.join(data_dir, graph_pattern)
        self.metadata_path = os.path.join(data_dir, metadata_file)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.graph_files = glob.glob(self.graph_pattern)
        self.mapping = self._match_files_to_metadata()

    def _match_files_to_metadata(self):
        mapping = {}
        for graph in self.graph_files:
            name = os.path.basename(graph).replace("_v3.h5", "")
            for record in self.metadata:
                if record["patient_id"] == name:
                    mapping[name] = {"file": graph, "meta": record}
        return mapping

    def load_training_nodes(self, patient_id, label, n_samples=10000):
        graph_file = self.mapping[patient_id]["file"]
        with h5py.File(graph_file, "r") as f:
            total_nodes = f["node_features"].shape[0]
            idx = np.sort(np.random.choice(total_nodes, min(n_samples, total_nodes), replace=False))

            return {
                "features": f["node_features"][idx],
                "coords": f["coordinates"][idx],
                "label": np.full(len(idx), label),
                "n_total": total_nodes
            }

    def load_full_graph(self, patient_id):
        graph_file = self.mapping[patient_id]["file"]
        with h5py.File(graph_file, "r") as f:
            return {
                "features": f["node_features"][:],
                "coords": f["coordinates"][:]
            }
