import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch_geometric.utils import k_hop_subgraph
import json
import glob
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

class NodeLevelGNN(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_layers=1):
        super(NodeLevelGNN, self).__init__()
        
        self.node_encoder = torch.nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            conv = SAGEConv(
                hidden_channels, 
                hidden_channels,
                normalize=False
            )
            self.convs.append(conv)
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels // 2, 1),
            torch.nn.Sigmoid()
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if module is self.classifier[-2]:
                    module.bias.data.fill_(-0.2)
                else:
                    module.bias.data.fill_(0.01)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x)
        
        if edge_attr is not None:
            edge_weight = self.edge_encoder(edge_attr).squeeze(-1)
        else:
            edge_weight = None
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if edge_weight is not None:
                x = conv.propagate(edge_index, x=x, size=None, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        node_predictions = self.classifier(x)
        return node_predictions

def create_graph_data(file_path, label=None):
    with h5py.File(file_path, 'r') as f:
        node_features = torch.FloatTensor(f['node_features'][:])
        edge_index = torch.LongTensor(f['edge_index'][:])
        edge_attr = torch.FloatTensor(f['edge_features'][:]) if 'edge_features' in f else None
        coordinates = torch.FloatTensor(f['coordinates'][:])
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=coordinates,
            y=torch.tensor([label] if label is not None else [], dtype=torch.float)
        )
        return data

def calculate_class_weights(node_counts):
    """Calculate class weights based on node counts in training data"""
    total_nodes = np.sum(node_counts)
    n_classes = len(node_counts)
    weights = total_nodes / (n_classes * node_counts)
    return torch.FloatTensor(weights)

def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        
        batch_size = data.batch[-1].item() + 1
        target = torch.zeros_like(out, device=device)
        
        for i in range(batch_size):
            mask = data.batch == i
            target[mask] = data.y[i]
        
        sample_weights = torch.ones_like(target, device=device)
        for i in range(len(class_weights)):
            sample_weights[target == i] = class_weights[i]
        
        loss = F.binary_cross_entropy(
            out.view(-1), 
            target.view(-1),
            weight=sample_weights.view(-1),
            reduction='mean'
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            
            batch_size = data.batch[-1].item() + 1
            for i in range(batch_size):
                mask = data.batch == i
                predictions.extend(out[mask].cpu().view(-1).tolist())
                labels.extend([data.y[i].item()] * mask.sum().item())
    
    return predictions, labels

def create_data_splits(graph_files, metadata, test_indices, val_ratio=0.2):
    """Create train and validation splits excluding test indices"""
    # Create mapping between metadata indices and graph files
    metadata_to_file = {}
    for i, meta_entry in enumerate(metadata):
        patient_id = meta_entry['patient_id']
        # Find the corresponding graph file
        for graph_file in graph_files:
            filename = os.path.basename(graph_file)
            file_patient_id = filename.split('_v3.h5')[0]
            if file_patient_id == patient_id:
                metadata_to_file[i] = graph_file
                break
    
    # Get available indices (excluding test set)
    available_indices = [i for i in range(len(metadata)) if i not in test_indices]
    
    # Separate indices by class
    class_0_indices = [i for i in available_indices if metadata[i]['label'] == 0]
    class_1_indices = [i for i in available_indices if metadata[i]['label'] == 1]
    
    # Shuffle both lists
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    # Calculate split sizes for each class
    val_size_0 = max(1, int(len(class_0_indices) * val_ratio))
    val_size_1 = max(1, int(len(class_1_indices) * val_ratio))
    
    # Split indices for each class
    val_indices = (class_0_indices[:val_size_0] + class_1_indices[:val_size_1])
    train_indices = (class_0_indices[val_size_0:] + class_1_indices[val_size_1:])
    
    # Create graph lists
    train_graphs = []
    val_graphs = []
    total_train_nodes = 0
    label_counts = np.zeros(2, dtype=int)
    
    # Process training graphs
    for idx in train_indices:
        if idx in metadata_to_file:
            graph = create_graph_data(metadata_to_file[idx], metadata[idx]['label'])
            sampled_graph = sample_from_graph(graph, n_samples_per_patient=20000)
            train_graphs.append(sampled_graph)
            total_train_nodes += sampled_graph.x.size(0)
            label_counts[int(metadata[idx]['label'])] += sampled_graph.x.size(0)
        else:
            print(f"Warning: No graph file found for metadata index {idx}")
    
    # Process validation graphs
    for idx in val_indices:
        if idx in metadata_to_file:
            graph = create_graph_data(metadata_to_file[idx], metadata[idx]['label'])
            sampled_graph = sample_from_graph(graph, n_samples_per_patient=20000)
            val_graphs.append(sampled_graph)
        else:
            print(f"Warning: No graph file found for metadata index {idx}")
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    print(f"Training class distribution: Class 0: {len([i for i in train_indices if metadata[i]['label'] == 0])}, " 
          f"Class 1: {len([i for i in train_indices if metadata[i]['label'] == 1])}")
    print(f"Validation class distribution: Class 0: {len([i for i in val_indices if metadata[i]['label'] == 0])}, "
          f"Class 1: {len([i for i in val_indices if metadata[i]['label'] == 1])}")
    
    return train_graphs, val_graphs, total_train_nodes, label_counts

def sample_from_graph(data, n_samples_per_patient=1000):
    total_nodes = data.x.size(0)
    sample_size = min(n_samples_per_patient, total_nodes)
    
    sampled_indices = torch.randperm(total_nodes)[:sample_size]
    
    k_hop = 2
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=sampled_indices, 
        num_hops=k_hop, 
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=total_nodes
    )
    
    sampled_x = data.x[subset]
    sampled_pos = data.pos[subset]
    sampled_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
    
    return Data(
        x=sampled_x,
        edge_index=edge_index,
        edge_attr=sampled_edge_attr,
        pos=sampled_pos,
        y=data.y
    )

def run_complete_analysis(metadata_file, graph_pattern, output_dir='gnn_analysis_results', cv_folds=None):
    """
    Run complete analysis with either LOO or k-fold CV
    
    Args:
        metadata_file: Path to metadata JSON file
        graph_pattern: Glob pattern for graph files
        output_dir: Output directory for output
        cv_folds: Number of CV folds (None for LOO)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    graph_files = glob.glob(graph_pattern)
    
    # Create mapping between graph files and metadata
    # Extract patient IDs from graph filenames and match with metadata
    file_to_metadata = {}
    metadata_to_file = {}
    
    for graph_file in graph_files:
        # Extract patient ID from filename (assuming format like "patient_id_v2.h5")
        filename = os.path.basename(graph_file)
        patient_id = filename.split('_v3.h5')[0]  # Remove the suffix
        
        # Find matching metadata entry
        matching_metadata = None
        for meta_entry in metadata:
            if meta_entry['patient_id'] == patient_id:
                matching_metadata = meta_entry
                break
        
        if matching_metadata is not None:
            file_to_metadata[graph_file] = matching_metadata
            metadata_to_file[matching_metadata['patient_id']] = graph_file
        else:
            print(f"Warning: No metadata found for graph file {graph_file}")
    
    # Filter metadata to only include patients with existing graph files
    available_metadata = [meta for meta in metadata if meta['patient_id'] in metadata_to_file]
    available_graph_files = list(file_to_metadata.keys())
    
    print(f"Found {len(available_metadata)} patients with both metadata and graph files")
    print(f"Available patients: {[meta['patient_id'] for meta in available_metadata]}")
    
    if len(available_metadata) == 0:
        raise ValueError("No matching metadata and graph files found!")
    
    all_summaries = []
    node_results = []
    
    # Setup cross-validation
    if cv_folds is None:
        # Leave-one-out
        fold_indices = [([i], f"LOO_{i}") for i in range(len(available_metadata))]
        print("Running leave-one-out cross-validation...")
    else:
        # K-fold CV
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_indices = [(test_idx, f"Fold_{i}") for i, (_, test_idx) in enumerate(kf.split(range(len(available_metadata))))]
        print(f"Running {cv_folds}-fold cross-validation...")
    
    for test_idx, fold_name in fold_indices:
        test_patients = [available_metadata[i] for i in test_idx]
        test_files = [metadata_to_file[p['patient_id']] for p in test_patients]
        
        print(f"\nTesting on {fold_name}")
        print(f"Test patients: {[p['patient_id'] for p in test_patients]}")
        print("Training model...")
        
        train_graphs, val_graphs, total_train_nodes, label_counts = create_data_splits(
            available_graph_files, 
            available_metadata, 
            test_idx
        )
        
        # Calculate class weights based on actual node distribution
        class_weights = calculate_class_weights(label_counts)
        
        print(f"Training with {total_train_nodes:,} nodes")
        print(f"Label distribution: {label_counts}")
        print(f"Class weights: {{0: {class_weights[0]:.3f}, 1: {class_weights[1]:.3f}}}")
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        
        sample_data = train_graphs[0]
        model = NodeLevelGNN(
            node_in_channels=sample_data.x.size(1),
            edge_in_channels=sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0,
            hidden_channels=64,
            num_layers=2
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50)
        
        best_val_auc = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
            
            val_predictions, val_labels = evaluate(model, val_loader, device)
            val_auc = roc_auc_score(val_labels, val_predictions)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step(val_auc)
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1:02d}, Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        model.load_state_dict(best_model_state)
        
        # Test on all patients in the test fold
        for test_patient, test_file in zip(test_patients, test_files):
            print(f"Predicting on test patient {test_patient['patient_id']}")
            test_graph = create_graph_data(test_file, test_patient['label'])
            print(f"Total nodes: {test_graph.x.size(0):,}")
            
            test_loader = DataLoader([test_graph], batch_size=1)
            predictions, labels = evaluate(model, test_loader, device)
            
            node_df = pd.DataFrame({
                'sample': test_patient['patient_id'],
                'X': test_graph.pos[:, 0],
                'Y': test_graph.pos[:, 1],
                'Z': test_graph.pos[:, 2],
                'pred': predictions,
                'label': test_patient['label'],
                'fold': fold_name
            })
            node_results.append(node_df)
            
            patient_pred = np.mean(predictions)
            summary = {
                'patient_id': test_patient['patient_id'],
                'label': test_patient['label'],
                'mean_prediction': patient_pred,
                'high_conf_ratio': np.mean(np.array(predictions) > 0.8),
                'n_nodes': len(predictions),
                'best_val_auc': best_val_auc,
                'fold': fold_name
            }
            all_summaries.append(summary)

    all_node_df = pd.concat(node_results, ignore_index=True)
    all_node_df.to_csv(f'{output_dir}/node_level_predictions2.csv', index=False)
    
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(f'{output_dir}/analysis_summary.csv', index=False)
    
    patient_predictions = summary_df['mean_prediction'].values
    patient_labels = summary_df['label'].values
    
    patient_auc = roc_auc_score(patient_labels, patient_predictions)
    patient_ap = average_precision_score(patient_labels, patient_predictions)
    
    # Plot overall ROC curve
    fpr, tpr, _ = roc_curve(patient_labels, patient_predictions)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'Overall AUC = {patient_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC Curve')
    plt.legend()
    plt.savefig(f'{output_dir}/patient_level_roc2.png')
    plt.close()
    
    # Print final output
    print("\nFinal Results:")
    print("-" * 40)
    print(f"Patient-Level Metrics:")
    print(f"Overall AUC: {patient_auc:.3f}")
    print(f"Average Precision: {patient_ap:.3f}")
    print(f"Mean Validation AUC: {summary_df['best_val_auc'].mean():.3f}")
    
    # Print per-fold metrics if using k-fold CV
    if cv_folds is not None:
        print("\nPer-fold metrics:")
        for fold in summary_df['fold'].unique():
            fold_df = summary_df[summary_df['fold'] == fold]
            fold_auc = roc_auc_score(fold_df['label'], fold_df['mean_prediction'])
            print(f"{fold} - AUC: {fold_auc:.3f}")
    
    return summary_df, all_node_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GNN end-to-end analysis.')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to directory containing metadata3.json and *_v3.h5 graph files.')
    parser.add_argument('--output_dir', type=str, default='gnn_analysis_results', 
                       help='Directory to save analysis output.')
    parser.add_argument('--cv_folds', type=int, default=None, 
                       help='Number of CV folds (set to None for LOO CV)')
    
    args = parser.parse_args()
    
    # Construct file paths from data directory
    metadata_file = os.path.join(args.data_dir, 'metadata3.json')
    graph_pattern = os.path.join(args.data_dir, '*_v3.h5')
    
    # Verify files exist
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    graph_files = glob.glob(graph_pattern)
    if not graph_files:
        raise FileNotFoundError(f"No graph files found matching pattern: {graph_pattern}")
    
    print(f"Found {len(graph_files)} graph files in {args.data_dir}")
    
    # Choose between LOO and k-fold CV
    cv_folds = args.cv_folds if args.cv_folds > 1 else None
    
    try:
        summary_df, node_df = run_complete_analysis(
            metadata_file, 
            graph_pattern, 
            args.output_dir,
            cv_folds=cv_folds
        )
        
        print("\nAnalysis completed successfully!")
        print(f"\nFiles generated in '{args.output_dir}' directory:")
        print("1. analysis_summary.csv - Patient-level output summary")
        print("2. node_level_predictions2.csv - Detailed node-level predictions")
        print("3. patient_level_roc2.png - Overall patient-level ROC curve")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
        
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
data = pd.read_csv("gnn_analysis_results/node_level_predictions2.csv")
def find_optimal_threshold(data, threshold_range=None, n_steps=100):
    """
    Find the optimal threshold that maximizes AUC score for patient-level predictions.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Must contain columns: 'sample', 'pred', 'label'
    threshold_range : tuple, optional
        (min_threshold, max_threshold) to search. Defaults to (data['pred'].min(), data['pred'].max())
    n_steps : int, optional
        Number of threshold values to try. Default is 100
    
    Returns:
    --------
    dict
        Contains 'optimal_threshold', 'best_auc', and 'threshold_results'
    """
    def calculate_patient_score(group, thresh):
        proportion_above_thresh = (group['pred'] >= thresh).mean()
        return pd.Series({
            'patient_score': proportion_above_thresh,
            'label': group['label'].iloc[0]
        })
    
    # Set default threshold range if none provided
    if threshold_range is None:
        threshold_range = (data['pred'].min(), data['pred'].max())
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    
    results = []
    for thresh in tqdm(thresholds, desc="Finding optimal threshold"):
        patient_scores = data.groupby('sample').apply(
            lambda x: calculate_patient_score(x, thresh)
        ).reset_index()
        
        current_auc = roc_auc_score(
            patient_scores['label'],
            patient_scores['patient_score']
        )
        
        results.append({
            'threshold': thresh,
            'auc': current_auc
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    best_idx = results_df['auc'].argmax()
    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_auc = results_df.loc[best_idx, 'auc']
    
    return {
        'optimal_threshold': optimal_threshold,
        'best_auc': best_auc,
        'threshold_results': results_df
    }

# Example usage:
result = find_optimal_threshold(
    data,
    threshold_range=(0.5, 1),  
    n_steps=20  
)

print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
print(f"Best AUC: {result['best_auc']:.3f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(result['threshold_results']['threshold'], 
         result['threshold_results']['auc'])
plt.xlabel('Threshold')
plt.ylabel('AUC')
plt.title('AUC vs Threshold')
plt.grid(True)
plt.show()
