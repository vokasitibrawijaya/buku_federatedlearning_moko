"""
run_heterogeneous.py - Simulasi FL dengan Heterogenitas Sistem

Script untuk menjalankan simulasi FL yang memodelkan:
1. System heterogeneity (klien lambat vs cepat)
2. Statistical heterogeneity (non-IID data)
3. Client dropout

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.model import SimpleCNN, CIFAR10CNN, get_parameters, set_parameters
from basic.data import (
    load_dataset,
    partition_iid,
    partition_dirichlet,
    get_client_dataloader,
    get_partition_stats
)
from basic.train import train, test


# Global variables
trainset = None
testset = None
client_indices = None
client_configs = None
device = None


def create_client_configs(
    num_clients: int,
    num_slow: int,
    num_unreliable: int,
    slow_factor: float = 2.0,
    dropout_prob: float = 0.3,
    seed: int = 42
) -> List[Dict]:
    """
    Membuat konfigurasi heterogenitas untuk setiap klien.
    
    Args:
        num_clients: Total klien
        num_slow: Jumlah klien lambat
        num_unreliable: Jumlah klien tidak stabil
        slow_factor: Faktor kelambatan (2.0 = 2x lebih lambat)
        dropout_prob: Probabilitas dropout
        seed: Random seed
    
    Returns:
        List of config dicts untuk setiap klien
    """
    random.seed(seed)
    
    configs = [{'type': 'fast', 'slow_factor': 1.0, 'dropout_prob': 0.0} 
               for _ in range(num_clients)]
    
    # Assign slow clients
    slow_indices = random.sample(range(num_clients), min(num_slow, num_clients))
    for idx in slow_indices:
        configs[idx]['type'] = 'slow'
        configs[idx]['slow_factor'] = slow_factor
    
    # Assign unreliable clients (from remaining fast clients)
    remaining = [i for i in range(num_clients) if configs[i]['type'] == 'fast']
    unreliable_indices = random.sample(
        remaining, 
        min(num_unreliable, len(remaining))
    )
    for idx in unreliable_indices:
        configs[idx]['type'] = 'unreliable'
        configs[idx]['dropout_prob'] = dropout_prob
    
    return configs


def get_heterogeneous_client_fn(
    dataset_name: str,
    batch_size: int,
    local_epochs: int,
    learning_rate: float
):
    """Factory untuk client_fn dengan heterogenitas."""
    global trainset, testset, client_indices, client_configs, device
    
    class HeterogeneousSimClient(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = int(cid)
            self.config = client_configs[self.cid]
            
            # Create model
            if dataset_name.lower() == 'mnist':
                self.model = SimpleCNN(num_classes=10)
            else:
                self.model = CIFAR10CNN(num_classes=10)
            
            self.model.to(device)
            
            # Create dataloader
            self.trainloader = get_client_dataloader(
                trainset,
                client_indices[self.cid],
                batch_size
            )
            self.testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False
            )
        
        def get_parameters(self, config):
            return get_parameters(self.model)
        
        def fit(self, parameters, config):
            # Check dropout
            if random.random() < self.config['dropout_prob']:
                # Simulate dropout by returning unchanged parameters
                print(f"[Client {self.cid}] DROPPED OUT!")
                # Return with very small weight so it doesn't affect aggregation much
                return (
                    parameters,  # Return original parameters unchanged
                    1,  # Minimal weight
                    {"dropped": True, "client_id": self.cid}
                )
            
            set_parameters(self.model, parameters)
            
            # Training
            start_time = time.time()
            
            history = train(
                self.model,
                self.trainloader,
                epochs=local_epochs,
                device=device,
                learning_rate=learning_rate
            )
            
            base_time = time.time() - start_time
            
            # Simulate slow client by reporting longer time
            # (In real scenario, training would actually take longer)
            effective_time = base_time * self.config['slow_factor']
            
            return (
                get_parameters(self.model),
                len(self.trainloader.dataset),
                {
                    "loss": history['loss'][-1],
                    "client_id": self.cid,
                    "client_type": self.config['type'],
                    "training_time": effective_time,
                    "dropped": False
                }
            )
        
        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            loss, accuracy = test(self.model, self.testloader, device)
            return (
                float(loss),
                len(self.testloader.dataset),
                {"accuracy": accuracy, "client_id": self.cid}
            )
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return HeterogeneousSimClient(cid)
    
    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average dengan handling untuk dropped clients."""
    if not metrics:
        return {}
    
    # Filter out dropped clients
    valid_metrics = [(n, m) for n, m in metrics if not m.get('dropped', False)]
    
    if not valid_metrics:
        return {}
    
    total_examples = sum(n for n, _ in valid_metrics)
    
    result = {}
    if 'accuracy' in valid_metrics[0][1]:
        acc = sum(n * m['accuracy'] for n, m in valid_metrics) / total_examples
        result['accuracy'] = acc
    
    return result


def run_heterogeneous_simulation(
    num_clients: int = 10,
    num_rounds: int = 20,
    dataset_name: str = 'mnist',
    partition: str = 'dirichlet',
    alpha: float = 0.3,
    fraction_fit: float = 0.5,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 42,
    # Heterogeneity parameters
    num_slow_clients: int = 3,
    num_unreliable_clients: int = 2,
    slow_factor: float = 2.0,
    dropout_prob: float = 0.3,
    # Output
    output_dir: str = './results_heterogeneous'
) -> Dict:
    """
    Menjalankan simulasi FL dengan heterogenitas.
    
    Args:
        num_clients: Jumlah klien
        num_rounds: Jumlah ronde
        dataset_name: Dataset
        partition: Tipe partisi
        alpha: Parameter Dirichlet
        fraction_fit: Fraksi klien per ronde
        local_epochs: Epoch lokal
        batch_size: Ukuran batch
        learning_rate: Learning rate
        seed: Random seed
        num_slow_clients: Jumlah klien lambat
        num_unreliable_clients: Jumlah klien tidak stabil
        slow_factor: Faktor kelambatan
        dropout_prob: Probabilitas dropout
        output_dir: Directory output
    
    Returns:
        Dictionary hasil eksperimen
    """
    global trainset, testset, client_indices, client_configs, device
    
    print("\n" + "="*60)
    print("HETEROGENEOUS FL SIMULATION")
    print("="*60)
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print(f"\nLoading {dataset_name.upper()} dataset...")
    trainset, testset = load_dataset(dataset_name, './data')
    
    # Partition
    print(f"Partitioning data ({partition}, alpha={alpha})...")
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    else:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    
    # Create client configs
    print("\nConfiguring client heterogeneity...")
    client_configs = create_client_configs(
        num_clients=num_clients,
        num_slow=num_slow_clients,
        num_unreliable=num_unreliable_clients,
        slow_factor=slow_factor,
        dropout_prob=dropout_prob,
        seed=seed
    )
    
    # Print summary
    print(f"\nClient Configuration Summary:")
    print(f"  Total clients: {num_clients}")
    print(f"  Fast clients: {sum(1 for c in client_configs if c['type'] == 'fast')}")
    print(f"  Slow clients: {sum(1 for c in client_configs if c['type'] == 'slow')}")
    print(f"  Unreliable clients: {sum(1 for c in client_configs if c['type'] == 'unreliable')}")
    print(f"  Slow factor: {slow_factor}x")
    print(f"  Dropout probability: {dropout_prob*100:.0f}%")
    
    # Initialize model
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN(num_classes=10)
    else:
        model = CIFAR10CNN(num_classes=10)
    
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    
    # Strategy
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=max(2, int(num_clients * fraction_fit * 0.5)),  # Lower minimum due to dropouts
        min_evaluate_clients=max(2, int(num_clients * fraction_fit * 0.5)),
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters
    )
    
    # Run simulation
    print(f"\nStarting simulation...")
    print("-"*60)
    
    history = start_simulation(
        client_fn=get_heterogeneous_client_fn(
            dataset_name,
            batch_size,
            local_epochs,
            learning_rate
        ),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Process results
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    results = {
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'dataset': dataset_name,
            'partition': partition,
            'alpha': alpha,
            'num_slow_clients': num_slow_clients,
            'num_unreliable_clients': num_unreliable_clients,
            'slow_factor': slow_factor,
            'dropout_prob': dropout_prob,
            'seed': seed
        },
        'client_configs': client_configs,
        'accuracies': [],
        'losses': [],
        'timestamp': timestamp
    }
    
    # Extract metrics
    if history.metrics_distributed:
        for round_num, metrics in history.metrics_distributed.get('accuracy', []):
            results['accuracies'].append({'round': round_num, 'accuracy': metrics})
    
    if history.losses_distributed:
        for round_num, loss in history.losses_distributed:
            results['losses'].append({'round': round_num, 'loss': loss})
    
    # Print final results
    if results['accuracies']:
        print(f"Final accuracy: {results['accuracies'][-1]['accuracy']*100:.2f}%")
    
    # Save results
    results_path = os.path.join(output_dir, f'results_heterogeneous_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot
    plot_heterogeneous_results(results, output_dir, timestamp)
    
    return results


def plot_heterogeneous_results(results: Dict, output_dir: str, timestamp: str):
    """Plot hasil simulasi heterogen."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    if results['losses']:
        rounds = [r['round'] for r in results['losses']]
        losses = [r['loss'] for r in results['losses']]
        axes[0].plot(rounds, losses, 'b-o', linewidth=2, markersize=5)
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if results['accuracies']:
        rounds = [r['round'] for r in results['accuracies']]
        accs = [r['accuracy'] * 100 for r in results['accuracies']]
        axes[1].plot(rounds, accs, 'g-o', linewidth=2, markersize=5)
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Test Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    # Client type distribution
    client_types = [c['type'] for c in results['client_configs']]
    type_counts = {
        'fast': client_types.count('fast'),
        'slow': client_types.count('slow'),
        'unreliable': client_types.count('unreliable')
    }
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[2].bar(type_counts.keys(), type_counts.values(), color=colors)
    axes[2].set_xlabel('Client Type')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Client Type Distribution')
    
    plt.tight_layout()
    
    config = results['config']
    fig.suptitle(
        f"Heterogeneous FL: {config['num_clients']} clients, "
        f"α={config['alpha']}, dropout={config['dropout_prob']*100:.0f}%",
        y=1.02
    )
    
    plot_path = os.path.join(output_dir, f'plot_heterogeneous_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Run Heterogeneous FL Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic config
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--num-rounds', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--partition', type=str, default='dirichlet')
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--fraction-fit', type=float, default=0.5)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    
    # Heterogeneity config
    parser.add_argument('--slow-clients', type=int, default=3)
    parser.add_argument('--unreliable-clients', type=int, default=2)
    parser.add_argument('--slow-factor', type=float, default=2.0)
    parser.add_argument('--dropout-prob', type=float, default=0.3)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results_heterogeneous')
    
    args = parser.parse_args()
    
    run_heterogeneous_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        dataset_name=args.dataset,
        partition=args.partition,
        alpha=args.alpha,
        fraction_fit=args.fraction_fit,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_slow_clients=args.slow_clients,
        num_unreliable_clients=args.unreliable_clients,
        slow_factor=args.slow_factor,
        dropout_prob=args.dropout_prob,
        output_dir=args.output_dir
    )
    
    print("\n✓ Simulation completed!")


if __name__ == "__main__":
    main()
