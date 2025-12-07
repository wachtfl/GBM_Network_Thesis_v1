class GNNEstimator:
    """
    Sklearn-compatible estimator wrapping a PyTorch Geometric GNN.
    """

    def __init__(
        self,
        node_in_channels,
        edge_in_channels,
        hidden_channels=64,
        num_layers=2,
        lr=1e-4,
        batch_size=4,
        max_epochs=30,
        sample_size=20000,
        device=None,
    ):
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.sample_size = sample_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = None    # classic models use scalar; GNN doesn’t, but keep API consistent.

    # --------------------------
    # --- REQUIRED BY SKLEARN ---
    # --------------------------
    def fit(self, graph_file_paths, labels):
        """
        graph_file_paths: list of H5 files, one per sample
        labels: shape (n_samples,)
        """
        # ==== Build model ====
        self.model = NodeLevelGNN(
            node_in_channels=self.node_in_channels,
            edge_in_channels=self.edge_in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ==== Convert input files -> PyG Data ====
        dataset = []
        for path, label in zip(graph_file_paths, labels):
            g = create_graph_data(path, label)
            g = sample_from_graph(g, n_samples_per_patient=self.sample_size)
            dataset.append(g)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # class weights based on node distribution
        node_counts = np.array([
            g.x.size(0) if g.y.item() == 0 else 0
            for g in dataset
        ])
        class_weights = calculate_class_weights([sum(labels == 0), sum(labels == 1)]).to(self.device)

        # ==== Training ====
        for epoch in range(self.max_epochs):
            loss = train_epoch(self.model, loader, optimizer, self.device, class_weights)

        return self

    def predict_proba(self, graph_file_paths):
        dataset = []
        for path in graph_file_paths:
            g = create_graph_data(path)
            g = sample_from_graph(g, n_samples_per_patient=self.sample_size)
            dataset.append(g)

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        preds, labs = evaluate(self.model, loader, self.device)
        preds = np.array(preds)

        # Return average prediction per graph
        # (classic models output 1 value per sample)
        return np.vstack([1 - preds.mean(), preds.mean()]).T

    def predict(self, graph_file_paths):
        proba = self.predict_proba(graph_file_paths)
        return (proba[:, 1] > 0.5).astype(int)
