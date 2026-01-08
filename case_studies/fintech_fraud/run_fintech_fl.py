"""
run_fintech_fl.py - Simulasi FL untuk Deteksi Fraud Fintech Indonesia

Studi Kasus: 10 fintech/bank digital berkolaborasi untuk deteksi fraud
tanpa membagikan data transaksi.

Konteks:
- OJK mengawasi 100+ fintech lending dan bank digital
- Data transaksi adalah aset sensitif
- Pola fraud lintas platform bisa dideteksi lebih baik dengan kolaborasi

Karakteristik:
- Extreme class imbalance (fraud 0.1-1%)
- Quantity skew antar platform
- Feature heterogeneity

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basic.model import SimpleMLP, get_parameters, set_parameters


# Konfigurasi 10 Fintech Indonesia
FINTECH_CONFIG = {
    0: {"name": "GoPay", "size": 50000, "fraud_ratio": 0.008},
    1: {"name": "OVO", "size": 40000, "fraud_ratio": 0.005},
    2: {"name": "Dana", "size": 30000, "fraud_ratio": 0.012},
    3: {"name": "ShopeePay", "size": 20000, "fraud_ratio": 0.003},
    4: {"name": "LinkAja", "size": 15000, "fraud_ratio": 0.007},
    5: {"name": "Kredivo", "size": 10000, "fraud_ratio": 0.010},
    6: {"name": "Akulaku", "size": 8000, "fraud_ratio": 0.004},
    7: {"name": "Jenius", "size": 5000, "fraud_ratio": 0.015},
    8: {"name": "Flip", "size": 3000, "fraud_ratio": 0.006},
    9: {"name": "Jago", "size": 2000, "fraud_ratio": 0.009}
}


def generate_synthetic_fraud_data(
    num_samples: int,
    fraud_ratio: float,
    num_features: int = 30,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection data.
    
    Mensimulasikan data transaksi dengan karakteristik:
    - Extreme class imbalance
    - Berbagai fitur transaksi
    
    Args:
        num_samples: Jumlah sampel
        fraud_ratio: Rasio fraud (0-1)
        num_features: Jumlah fitur
        seed: Random seed
    
    Returns:
        Tuple (X, y) features dan labels
    """
    np.random.seed(seed)
    
    # Generate imbalanced classification data
    n_fraud = max(int(num_samples * fraud_ratio), 10)
    n_normal = num_samples - n_fraud
    
    # Normal transactions
    X_normal = np.random.randn(n_normal, num_features)
    # Add some realistic patterns
    X_normal[:, 0] = np.random.uniform(10, 1000, n_normal)  # Amount
    X_normal[:, 1] = np.random.choice([0, 1], n_normal, p=[0.7, 0.3])  # International
    X_normal[:, 2] = np.random.uniform(0, 23, n_normal)  # Hour
    
    # Fraud transactions (slightly different distribution)
    X_fraud = np.random.randn(n_fraud, num_features)
    X_fraud[:, 0] = np.random.uniform(500, 10000, n_fraud)  # Higher amounts
    X_fraud[:, 1] = np.random.choice([0, 1], n_fraud, p=[0.3, 0.7])  # More international
    X_fraud[:, 2] = np.random.uniform(0, 5, n_fraud)  # Late night hours
    X_fraud[:, 3:6] = X_fraud[:, 3:6] + 2  # Anomalous patterns
    
    # Combine
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0] * n_normal + [1] * n_fraud)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X.astype(np.float32), y.astype(np.int64)


def create_fintech_datasets(
    fintech_config: Dict,
    num_features: int = 30,
    seed: int = 42
) -> Tuple[List[TensorDataset], TensorDataset]:
    """
    Membuat dataset untuk setiap fintech.
    
    Args:
        fintech_config: Konfigurasi per fintech
        num_features: Jumlah fitur
        seed: Random seed
    
    Returns:
        Tuple (list of train datasets, test dataset)
    """
    np.random.seed(seed)
    
    train_datasets = []
    test_X_all = []
    test_y_all = []
    
    for fintech_id in range(len(fintech_config)):
        config = fintech_config[fintech_id]
        
        # Generate data untuk fintech ini
        X, y = generate_synthetic_fraud_data(
            num_samples=config["size"],
            fraud_ratio=config["fraud_ratio"],
            num_features=num_features,
            seed=seed + fintech_id
        )
        
        # Split train/test (90/10)
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create dataset
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_datasets.append(train_dataset)
        
        test_X_all.append(X_test)
        test_y_all.append(y_test)
        
        # Stats
        n_fraud_train = y_train.sum()
        print(f"Fintech {fintech_id} ({config['name']}): "
              f"{len(X_train)} samples, {n_fraud_train} fraud ({n_fraud_train/len(X_train)*100:.2f}%)")
    
    # Combined test set
    test_X = np.vstack(test_X_all)
    test_y = np.concatenate(test_y_all)
    test_dataset = TensorDataset(
        torch.FloatTensor(test_X),
        torch.LongTensor(test_y)
    )
    
    print(f"\nTotal test samples: {len(test_X)}, fraud: {test_y.sum()} ({test_y.sum()/len(test_y)*100:.2f}%)")
    
    return train_datasets, test_dataset


# Global variables
train_datasets = None
test_dataset = None
device = None


def get_fintech_client_fn(
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    num_features: int = 30
):
    """Factory untuk fintech clients."""
    global train_datasets, test_dataset, device
    
    class FintechClient(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = int(cid)
            self.fintech_name = FINTECH_CONFIG[self.cid]["name"]
            
            # MLP for fraud detection
            self.model = SimpleMLP(
                input_dim=num_features,
                hidden_dims=[64, 32, 16],
                num_classes=2
            )
            self.model.to(device)
            
            self.trainloader = DataLoader(
                train_datasets[self.cid],
                batch_size=batch_size,
                shuffle=True
            )
            self.testloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        def get_parameters(self, config):
            return get_parameters(self.model)
        
        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            
            # Training dengan class weighting untuk handle imbalance
            self.model.train()
            
            # Calculate class weights
            labels = [y.item() for _, y in train_datasets[self.cid]]
            n_fraud = sum(labels)
            n_normal = len(labels) - n_fraud
            if n_fraud > 0:
                weight_fraud = n_normal / n_fraud
            else:
                weight_fraud = 1.0
            
            class_weights = torch.FloatTensor([1.0, weight_fraud]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
            
            total_loss = 0.0
            for epoch in range(local_epochs):
                for X_batch, y_batch in self.trainloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            avg_loss = total_loss / (len(self.trainloader) * local_epochs)
            
            return (
                get_parameters(self.model),
                len(train_datasets[self.cid]),
                {"loss": avg_loss, "fintech_id": self.cid}
            )
        
        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            self.model.eval()
            
            all_preds = []
            all_labels = []
            all_probs = []
            total_loss = 0.0
            
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for X_batch, y_batch in self.testloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = (probs > 0.5).long()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Calculate metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            accuracy = (all_preds == all_labels).mean()
            
            # Handle edge cases
            if len(np.unique(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, all_probs)
            else:
                auc_roc = 0.0
            
            if all_preds.sum() > 0:
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
            
            return (
                float(total_loss / len(self.testloader)),
                len(test_dataset),
                {
                    "accuracy": accuracy,
                    "auc_roc": auc_roc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fintech_id": self.cid
                }
            )
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return FintechClient(cid)
    
    return client_fn


def weighted_average_fraud(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average untuk fraud detection metrics."""
    if not metrics:
        return {}
    
    total_examples = sum(n for n, _ in metrics)
    
    result = {}
    metric_names = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']
    
    for name in metric_names:
        if name in metrics[0][1]:
            weighted_sum = sum(n * m.get(name, 0) for n, m in metrics)
            result[name] = weighted_sum / total_examples
    
    return result


def run_fintech_simulation(
    num_rounds: int = 30,
    fraction_fit: float = 0.7,
    local_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    num_features: int = 30,
    seed: int = 42,
    output_dir: str = './results_fintech'
) -> Dict:
    """
    Jalankan simulasi FL untuk deteksi fraud fintech.
    
    Args:
        num_rounds: Jumlah ronde
        fraction_fit: Fraksi fintech per ronde
        local_epochs: Epoch lokal
        batch_size: Ukuran batch
        learning_rate: Learning rate
        num_features: Jumlah fitur
        seed: Random seed
        output_dir: Directory output
    
    Returns:
        Dictionary hasil
    """
    global train_datasets, test_dataset, device
    
    print("\n" + "="*70)
    print("SIMULASI FL: DETEKSI FRAUD FINTECH INDONESIA")
    print("Kolaborasi 10 Fintech/Bank Digital")
    print("="*70)
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate data
    print("\nMembuat synthetic fraud data...")
    train_datasets, test_dataset = create_fintech_datasets(
        FINTECH_CONFIG,
        num_features=num_features,
        seed=seed
    )
    
    # Initialize model
    model = SimpleMLP(
        input_dim=num_features,
        hidden_dims=[64, 32, 16],
        num_classes=2
    )
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    
    # Strategy
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
    
    num_fintechs = len(FINTECH_CONFIG)
    min_clients = max(2, int(num_fintechs * fraction_fit))
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,  # Evaluate all
        min_fit_clients=min_clients,
        min_evaluate_clients=num_fintechs,
        min_available_clients=num_fintechs,
        evaluate_metrics_aggregation_fn=weighted_average_fraud,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters
    )
    
    # Run simulation
    print(f"\nMemulai simulasi dengan {num_fintechs} fintech, {num_rounds} ronde...")
    print("-"*70)
    
    history = start_simulation(
        client_fn=get_fintech_client_fn(
            batch_size, local_epochs, learning_rate, num_features
        ),
        num_clients=num_fintechs,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Process results
    print("\n" + "="*70)
    print("SIMULASI SELESAI")
    print("="*70)
    
    results = {
        'config': {
            'scenario': 'fintech_fraud_detection',
            'num_fintechs': num_fintechs,
            'num_rounds': num_rounds,
            'seed': seed
        },
        'fintechs': FINTECH_CONFIG,
        'metrics_per_round': [],
        'timestamp': timestamp
    }
    
    # Extract metrics
    if history.metrics_distributed:
        for metric_name in ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']:
            if metric_name in history.metrics_distributed:
                for round_num, value in history.metrics_distributed[metric_name]:
                    # Find or create entry for this round
                    round_entry = next(
                        (r for r in results['metrics_per_round'] if r['round'] == round_num),
                        None
                    )
                    if round_entry is None:
                        round_entry = {'round': round_num}
                        results['metrics_per_round'].append(round_entry)
                    round_entry[metric_name] = value
    
    # Sort by round
    results['metrics_per_round'].sort(key=lambda x: x['round'])
    
    # Print final metrics
    print("\n" + "-"*70)
    print("HASIL AKHIR")
    print("-"*70)
    
    if results['metrics_per_round']:
        final = results['metrics_per_round'][-1]
        print(f"\nMetrik pada ronde terakhir:")
        print(f"  AUC-ROC:    {final.get('auc_roc', 0)*100:.2f}%")
        print(f"  Precision:  {final.get('precision', 0)*100:.2f}%")
        print(f"  Recall:     {final.get('recall', 0)*100:.2f}%")
        print(f"  F1-Score:   {final.get('f1', 0)*100:.2f}%")
        
        print("\nPerbandingan (estimasi):")
        print("┌─────────────────┬──────────┬───────────┬────────┬──────┐")
        print("│ Metode          │ AUC-ROC  │ Precision │ Recall │  F1  │")
        print("├─────────────────┼──────────┼───────────┼────────┼──────┤")
        print("│ Lokal (avg)     │   ~82%   │    ~15%   │  ~70%  │ ~25% │")
        print(f"│ Federated (ini) │   {final.get('auc_roc', 0)*100:5.1f}%  │   {final.get('precision', 0)*100:5.1f}%  │ {final.get('recall', 0)*100:5.1f}% │{final.get('f1', 0)*100:5.1f}%│")
        print("│ Centralized*    │   ~94%   │    ~45%   │  ~85%  │ ~59% │")
        print("└─────────────────┴──────────┴───────────┴────────┴──────┘")
        print("*Jika data bisa dikumpulkan (melanggar privasi)")
    
    # Save results
    results_path = os.path.join(output_dir, f'results_fintech_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nHasil disimpan di: {results_path}")
    
    # Plot
    plot_fintech_results(results, output_dir, timestamp)
    
    return results


def plot_fintech_results(results: Dict, output_dir: str, timestamp: str):
    """Plot hasil simulasi fintech."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = results['metrics_per_round']
    if metrics:
        rounds = [m['round'] for m in metrics]
        
        # AUC-ROC
        auc_rocs = [m.get('auc_roc', 0) * 100 for m in metrics]
        axes[0].plot(rounds, auc_rocs, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Ronde')
        axes[0].set_ylabel('AUC-ROC (%)')
        axes[0].set_title('AUC-ROC per Ronde')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])
        
        # Precision, Recall, F1
        precisions = [m.get('precision', 0) * 100 for m in metrics]
        recalls = [m.get('recall', 0) * 100 for m in metrics]
        f1s = [m.get('f1', 0) * 100 for m in metrics]
        
        axes[1].plot(rounds, precisions, 'r-o', label='Precision', linewidth=2, markersize=4)
        axes[1].plot(rounds, recalls, 'g-s', label='Recall', linewidth=2, markersize=4)
        axes[1].plot(rounds, f1s, 'b-^', label='F1', linewidth=2, markersize=4)
        axes[1].set_xlabel('Ronde')
        axes[1].set_ylabel('Score (%)')
        axes[1].set_title('Precision, Recall, F1')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    # Fintech data distribution
    fintechs = results['fintechs']
    names = [fintechs[str(i)]["name"] for i in range(len(fintechs))]
    sizes = [fintechs[str(i)]["size"] / 1000 for i in range(len(fintechs))]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    bars = axes[2].barh(names, sizes, color=colors)
    axes[2].set_xlabel('Jumlah Transaksi (ribu)')
    axes[2].set_title('Data per Fintech')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    
    fig.suptitle(
        "Simulasi FL: Deteksi Fraud 10 Fintech Indonesia",
        y=1.02, fontsize=12, fontweight='bold'
    )
    
    plot_path = os.path.join(output_dir, f'plot_fintech_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot disimpan di: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Simulasi FL Deteksi Fraud Fintech',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-rounds', type=int, default=30)
    parser.add_argument('--fraction-fit', type=float, default=0.7)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./results_fintech')
    
    args = parser.parse_args()
    
    run_fintech_simulation(
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n✓ Simulasi selesai!")


if __name__ == "__main__":
    main()
