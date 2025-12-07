from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from models.gnns.gnn_estimator_model import GNNEstimator


class ModelFactory:
    @staticmethod
    def get(model_name):
        if model_name == "logistic":
            return LogisticRegression()
        if model_name == "dtree":
            return DecisionTreeClassifier()
        if model_name == "rf":
            return RandomForestClassifier()
        if model_name == "svm":
            return SVC()
        if model_name == "gnn":
            return GNNEstimator(**kwargs)
        raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def create(model_name, config):
        model_name = model_name.lower()

        if model_name == "rf":
            return RandomForestClassifier(**config)

        if model_name == "gnn":
            return GNNEstimator(
                node_in_channels=config["node_in_channels"],
                edge_in_channels=config["edge_in_channels"],
                hidden_channels=config.get("hidden_channels", 64),
                num_layers=config.get("num_layers", 2),
                lr=config.get("lr", 1e-4),
                max_epochs=config.get("max_epochs", 30),
                batch_size=config.get("batch_size", 4),
                sample_size=config.get("sample_size", 20000),
            )

        raise ValueError(f"Unknown model type: {model_name}")
