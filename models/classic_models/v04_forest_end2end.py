import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier ## you can try and play with other models
from sklearn.preprocessing import StandardScaler
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import json
import glob
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

def collect_training_data(file_path, label, n_samples=10000):
    """Collect training data from a single graph"""
    with h5py.File(file_path, 'r') as f:
        total_nodes = f['node_features'].shape[0]
        sample_size = min(n_samples, total_nodes)
        indices = np.sort(np.random.choice(total_nodes, sample_size, replace=False))
        
        features = f['node_features'][indices]
        labels = np.full(sample_size, label)
        
        return {
            'features': features,
            'labels': labels,
            'total_nodes': total_nodes
        }

def train_model(training_data):
    """Train model using collected data with class weights"""
    # Combine all features and labels
    all_features = np.vstack([data['features'] for data in training_data])
    all_labels = np.concatenate([data['labels'] for data in training_data])
    
    print(f"Training with {len(all_features)} nodes")
    label_counts = np.bincount(all_labels.astype(int))
    print(f"Label distribution: {label_counts}")
    
    # Calculate class weights
    n_samples = sum(label_counts)
    weights = {
        0: n_samples / (2 * label_counts[0]),
        1: n_samples / (2 * label_counts[1])
    }
    print(f"Class weights: {weights}")
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    
    # Train model with weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight=weights,
        random_state=42,
        n_jobs=-1
    )
    model.fit(scaled_features, all_labels)
    
    return model, scaler

def process_graph(file_path, patient_id, model, scaler, batch_size=10000):
    """Process a single graph and return predictions with coordinates"""
    all_predictions = []
    node_coordinates = []
    
    with h5py.File(file_path, 'r') as f:
        total_nodes = f['node_features'].shape[0]
        print(f"Total nodes: {total_nodes:,}")
        
        for start_idx in tqdm(range(0, total_nodes, batch_size), desc="Processing nodes"):
            end_idx = min(start_idx + batch_size, total_nodes)
            
            features = f['node_features'][start_idx:end_idx]
            coords = f['coordinates'][start_idx:end_idx]
            
            scaled_features = scaler.transform(features)
            predictions = model.predict_proba(scaled_features)[:, 1]
            
            all_predictions.extend(predictions)
            node_coordinates.extend(coords)
    
    return np.array(all_predictions), np.array(node_coordinates)

def run_complete_analysis(metadata_file, graph_pattern, output_dir='analysis_results'):
    """Run complete analysis pipeline with leave-one-out cross-validation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata and get graph files
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    graph_files = glob.glob(graph_pattern)
    graph_files = glob.glob(graph_pattern)

    # Create mapping between graph files and metadata
    # Extract patient IDs from graph filenames and match with metadata
    file_to_metadata = {}
    metadata_to_file = {}
    
    for graph_file in graph_files:
        # Extract patient ID from filename (assuming format like "patient_id_v3.h5")
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
    patient_predictions = []
    patient_labels = []
    node_results = []
    
    # LOO-CV
    print("Running leave-one-out cross-validation...")
    for test_idx in range(len(available_metadata)):
        test_patient = available_metadata[test_idx]
        test_file = metadata_to_file[test_patient['patient_id']]
        print(f"\nTesting on patient {test_patient['patient_id']}")
        
        ### Collect training data from all OTHER patients
        training_data = []
        for train_idx in range(len(available_metadata)):
            if train_idx != test_idx:
                train_patient = available_metadata[train_idx]
                train_file = metadata_to_file[train_patient['patient_id']]
                train_data = collect_training_data(
                    train_file,
                    train_patient['label']
                )
                training_data.append(train_data)
        
        print("Training model...")
        model, scaler = train_model(training_data)
        
        print(f"Predicting on test patient {test_patient['patient_id']}")
        predictions, coordinates = process_graph(test_file, test_patient['patient_id'], model, scaler)
    
        for pred, coord in zip(predictions, coordinates):
            node_results.append({
                'sample': test_patient['patient_id'],
                'X': coord[0],
                'Y': coord[1],
                'Z': coord[2],
                'pred': pred,
                'label': test_patient['label']
            })
        ## plot the preds dists per test patient
        plt.figure(figsize=(8, 4))
        sns.histplot(predictions, bins=50)
        plt.title(f'Node Prediction Distribution - Patient {test_patient["patient_id"]}')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/pred_dist_{test_patient["patient_id"]}.png')
        plt.close()
        
        patient_pred = np.mean(predictions)
        patient_predictions.append(patient_pred)
        patient_labels.append(test_patient['label'])
        summary = {
            'patient_id': test_patient['patient_id'],
            'label': test_patient['label'],
            'mean_prediction': patient_pred,
            'high_conf_ratio': np.mean(predictions > 0.8),
            'n_nodes': len(predictions)
        }
        all_summaries.append(summary)
    
    node_df = pd.DataFrame(node_results)
    node_df.to_csv(f'{output_dir}/node_level_predictions2.csv', index=False)
    
    patient_auc = roc_auc_score(patient_labels, patient_predictions)
    patient_ap = average_precision_score(patient_labels, patient_predictions)
    
    fpr, tpr, _ = roc_curve(patient_labels, patient_predictions)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {patient_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC Curve')
    plt.legend()
    plt.savefig(f'{output_dir}/patient_level_roc.png')
    plt.close()
    
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(f'{output_dir}/analysis_summary.csv', index=False)
    
    print("\nFinal Results:")
    print("-" * 40)
    print(f"Patient-Level Metrics:")
    print(f"Average Precision: {patient_ap:.3f}")
    
    return summary_df, node_df


def find_optimal_threshold(data, threshold_range=None, n_steps=100):
    """
    Find the optimal threshold that maximizes AUC score for patient-level predictions.
    """
    def calculate_patient_score(group, thresh):
        proportion_above_thresh = (group['pred'] >= thresh).mean()
        return pd.Series({
            'patient_score': proportion_above_thresh,
            'label': group['label'].iloc[0]
        })

    if threshold_range is None:
        threshold_range = (data['pred'].min(), data['pred'].max())

    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    
    results = []
    for thresh in tqdm(thresholds, desc="Finding optimal threshold"):
        patient_scores = data.groupby('sample').apply(
            lambda x: calculate_patient_score(x, thresh)
        ).reset_index()
        
        # Calculate AUC for current threshold
        current_auc = roc_auc_score(
            patient_scores['label'],
            patient_scores['patient_score']
        )
        
        results.append({
            'threshold': thresh,
            'auc': current_auc
        })
    results_df = pd.DataFrame(results)
    best_idx = results_df['auc'].argmax()
    optimal_threshold = results_df.loc[best_idx, 'threshold']
    best_auc = results_df.loc[best_idx, 'auc']
    
    return {
        'optimal_threshold': optimal_threshold,
        'best_auc': best_auc,
        'threshold_results': results_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run forest end-to-end analysis.')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to directory containing metadata2.json and *_v3.h5 graph files.')
    parser.add_argument('--output_dir', type=str, default='forest_analysis_results', 
                       help='Directory to save analysis output.')
    
    args = parser.parse_args()
    
    # Construct file paths from data directory
    metadata_file = os.path.join(args.data_dir, 'metadata2.json')
    graph_pattern = os.path.join(args.data_dir, '*_v3.h5')
    
    # Verify files exist
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    graph_files = glob.glob(graph_pattern)
    if not graph_files:
        raise FileNotFoundError(f"No graph files found matching pattern: {graph_pattern}")
    
    print(f"Found {len(graph_files)} graph files in {args.data_dir}")
    
    try:
        summary_df, node_df = run_complete_analysis(metadata_file, graph_pattern, args.output_dir)
        print("\nAnalysis completed successfully!")
        print(f"\nFiles generated in '{args.output_dir}' directory:")
        print("1. analysis_summary.csv - Complete output summary")
        print("2. node_level_predictions2.csv - Detailed node-level predictions with coordinates")
        print("3. pred_dist_*.png - Prediction distributions")
        print("4. patient_level_roc.png - Overall patient-level ROC curve")
        
        # Run threshold optimization
        result = find_optimal_threshold(
            node_df,
            threshold_range=(0.5, 1),  # Optional: specify range to search
            n_steps=50  # Optional: specify number of thresholds to try
        )

        print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
        print(f"Best AUC: {result['best_auc']:.3f}")
        
        # Save threshold analysis plot
        plt.figure(figsize=(10, 6))
        plt.plot(result['threshold_results']['threshold'], 
                 result['threshold_results']['auc'])
        plt.xlabel('Threshold')
        plt.ylabel('AUC')
        plt.title('AUC vs Threshold')
        plt.grid(True)
        plt.savefig(f'{args.output_dir}/threshold_analysis.png')
        plt.close()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e