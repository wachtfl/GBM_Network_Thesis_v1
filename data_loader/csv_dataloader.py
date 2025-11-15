import pandas as pd

class AggregatedFeatureLoader:
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)

    def get_features_and_labels(self):
        sample_names = self.df["graph_base"]
        X = self.df.drop(["label", "graph_base"], axis=1)
        y = self.df["label"]
        return X, y, sample_names
