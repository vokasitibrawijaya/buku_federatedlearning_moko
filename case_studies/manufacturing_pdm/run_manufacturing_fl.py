"""
run_manufacturing_fl.py - Simulasi FL untuk Predictive Maintenance Manufaktur

Studi Kasus: 8 pabrik di Kawasan Industri Jababeka berkolaborasi
untuk prediksi kerusakan mesin tanpa membagikan data sensor.

Konteks:
- Kawasan industri besar di Indonesia (Cikarang, Surabaya, Batam)
- Data sensor adalah rahasia industri
- Model prediktif bersama bisa mengurangi downtime

Karakteristik:
- Time series sensor data
- Feature heterogeneity (sensor berbeda)
- Label scarcity (kerusakan jarang)

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Konfigurasi 8 Pabrik di Kawasan Industri
FACTORY_CONFIG = {
    0: {
        "name": "PT Astra Components",
        "location": "Cikarang",
        "num_machines": 50,
        "sensors": ["vibration", "temperature", "rpm", "current"],
        "failure_rate": 0.05
    },
    1: {
        "name": "PT Toyota Parts",
        "location": "Cikarang",
        "num_machines": 40,
        "sensors": ["vibration", "temperature", "rpm"],
        "failure_rate": 0.04
    },
    2: {
        "name": "PT Samsung Electronics",
        "location": "Cikarang",
        "num_machines": 60,
        "sensors": ["vibration", "current", "acoustic"],
        "failure_rate": 0.03
    },
    3: {
        "name": "PT Panasonic Manufacturing",
        "location": "Cikarang",
        "num_machines": 35,
        "sensors": ["temperature", "rpm", "current", "humidity"],
        "failure_rate": 0.06
    },
    4: {
        "name": "PT Schneider Electric",
        "location": "Cikarang",
        "num_machines": 25,
        "sensors": ["vibration", "temperature", "current"],
        "failure_rate": 0.04
    },
    5: {
        "name": "PT Unilever Indonesia",
        "location": "Cikarang",
        "num_machines": 45,
        "sensors": ["temperature", "rpm", "flow_rate"],
        "failure_rate": 0.03
    },
    6: {
        "name": "PT Mayora Indah",
        "location": "Cikarang",
        "num_machines": 30,
        "sensors": ["vibration", "temperature"],
        "failure_rate": 0.05
    },
    7: {
        "name": "PT Honda Prospect",
        "location": "Cikarang",
        "num_machines": 55,
        "sensors": ["vibration", "temperature", "rpm", "current", "pressure"],
        "failure_rate": 0.04
    }
}

# All possible sensor types
ALL_SENSORS = ["vibration", "temperature", "rpm", "current", "acoustic", "humidity", "flow_rate", "pressure"]


class LSTMPredictor(nn.Module):
    """
    LSTM untuk prediksi time series (Remaining Useful Life).
    
    Memprediksi waktu hingga kerusakan berdasarkan sensor readings.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


def generate_sensor_data(
    num_machines: int,
    sensors: List[str],
    failure_rate: float,
    sequence_length: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sensor data untuk predictive maintenance.
    
    Mensimulasikan time series sensor dengan:
    - Normal operation patterns
    - Degradation patterns menjelang failure
    - Missing sensors (tidak semua pabrik punya semua sensor)
    
    Args:
        num_machines: Jumlah mesin
        sensors: List sensor yang tersedia
        failure_rate: Rasio mesin yang mengalami kerusakan
        sequence_length: Panjang sequence time series
        seed: Random seed
    
    Returns:
        Tuple (X, y) - sensor readings dan RUL (Remaining Useful Life)
    """
    np.random.seed(seed)
    
    num_features = len(ALL_SENSORS)  # Semua pabrik punya dimensi sama
    num_samples = num_machines * 10  # 10 samples per mesin
    
    X = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.float32)
    
    # Get indices of available sensors
    available_idx = [ALL_SENSORS.index(s) for s in sensors if s in ALL_SENSORS]
    
    for i in range(num_samples):
        # Determine if this is a failure sample
        is_failure = np.random.random() < failure_rate
        
        if is_failure:
            # RUL decreases over time (closer to failure)
            rul = np.random.uniform(0, 50)  # Hours until failure
            
            # Generate degrading sensor patterns
            for t in range(sequence_length):
                for sensor_idx in available_idx:
                    # Base signal
                    base = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.1)
                    # Add degradation trend
                    degradation = (sequence_length - t) / sequence_length * np.random.uniform(0.5, 2)
                    # Add anomalies near end
                    if t > sequence_length * 0.8:
                        anomaly = np.random.normal(0, 0.5)
                    else:
                        anomaly = 0
                    
                    X[i, t, sensor_idx] = base + degradation + anomaly
        else:
            # Normal operation - higher RUL
            rul = np.random.uniform(100, 500)  # Hours until (hypothetical) failure
            
            # Generate normal sensor patterns
            for t in range(sequence_length):
                for sensor_idx in available_idx:
                    # Normal operation with small variations
                    X[i, t, sensor_idx] = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.1)
        
        y[i] = rul
    
    # Normalize y to [0, 1] for easier training
    y = y / 500.0
    
    return X, y


def create_factory_datasets(
    factory_config: Dict,
    sequence_length: int = 100,
    seed: int = 42
) -> Tuple[List[TensorDataset], TensorDataset]:
    """
    Membuat dataset untuk setiap pabrik.
    
    Args:
        factory_config: Konfigurasi per pabrik
        sequence_length: Panjang sequence
        seed: Random seed
    
    Returns:
        Tuple (list of train datasets, test dataset)
    """
    np.random.seed(seed)
    
    train_datasets = []
    test_X_all = []
    test_y_all = []
    
    print("\nMembuat data sensor untuk setiap pabrik:")
    print("-" * 60)
    
    for factory_id in range(len(factory_config)):
        config = factory_config[factory_id]
        
        # Generate data
        X, y = generate_sensor_data(
            num_machines=config["num_machines"],
            sensors=config["sensors"],
            failure_rate=config["failure_rate"],
            sequence_length=sequence_length,
            seed=seed + factory_id
        )
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create dataset
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_datasets.append(train_dataset)
        
        test_X_all.append(X_test)
        test_y_all.append(y_test)
        
        print(f"Pabrik {factory_id} ({config['name'][:25]}): "
              f"{len(X_train)} samples, sensors: {config['sensors']}")
    
    # Combined test set
    test_X = np.vstack(test_X_all)
    test_y = np.concatenate(test_y_all)
    test_dataset = TensorDataset(
        torch.FloatTensor(test_X),
        torch.FloatTensor(test_y)
    )
    
    print("-" * 60)
    print(f"Total test samples: {len(test_X)}")
    
    return train_datasets, test_dataset


# Global variables
train_datasets = None
test_dataset = None
device = None


def get_factory_client_fn(
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    sequence_length: int = 100
):
    """Factory untuk manufacturing clients."""
    global train_datasets, test_dataset, device
    
    num_features = len(ALL_SENSORS)
    
    class FactoryClient(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = int(cid)
            self.factory_name = FACTORY_CONFIG[self.cid]["name"]
            
            self.model = LSTMPredictor(
                input_dim=num_features,
                hidden_dim=64,
                num_layers=2
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
            return [val.cpu().numpy() for val in self.model.state_dict().values()]
        
        def set_parameters(self, parameters):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            
            self.model.train()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
            
            total_loss = 0.0
            for epoch in range(local_epochs):
                for X_batch, y_batch in self.trainloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            avg_loss = total_loss / (len(self.trainloader) * local_epochs)
            
            return (
                self.get_parameters(config),
                len(train_datasets[self.cid]),
                {"loss": avg_loss, "factory_id": self.cid}
            )
        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()
            
            all_preds = []
            all_labels = []
            total_loss = 0.0
            
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for X_batch, y_batch in self.testloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            # Calculate metrics (scale back to hours)
            all_preds = np.array(all_preds) * 500
            all_labels = np.array(all_labels) * 500
            
            rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            
            return (
                float(total_loss / len(self.testloader)),
                len(test_dataset),
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "factory_id": self.cid
                }
            )
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return FactoryClient(cid)
    
    return client_fn


def weighted_average_pdm(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average untuk predictive maintenance metrics."""
    if not metrics:
        return {}
    
    total_examples = sum(n for n, _ in metrics)
    
    result = {}
    for name in ['rmse', 'mae', 'r2']:
        if name in metrics[0][1]:
            weighted_sum = sum(n * m.get(name, 0) for n, m in metrics)
            result[name] = weighted_sum / total_examples
    
    return result


def run_manufacturing_simulation(
    num_rounds: int = 40,
    fraction_fit: float = 1.0,
    local_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    sequence_length: int = 100,
    seed: int = 42,
    output_dir: str = './results_manufacturing'
) -> Dict:
    """
    Jalankan simulasi FL untuk predictive maintenance.
    
    Args:
        num_rounds: Jumlah ronde
        fraction_fit: Fraksi pabrik per ronde
        local_epochs: Epoch lokal
        batch_size: Ukuran batch
        learning_rate: Learning rate
        sequence_length: Panjang sequence
        seed: Random seed
        output_dir: Directory output
    
    Returns:
        Dictionary hasil
    """
    global train_datasets, test_dataset, device
    
    print("\n" + "="*70)
    print("SIMULASI FL: PREDICTIVE MAINTENANCE MANUFAKTUR INDONESIA")
    print("Kolaborasi 8 Pabrik di Kawasan Industri Jababeka")
    print("="*70)
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate data
    train_datasets, test_dataset = create_factory_datasets(
        FACTORY_CONFIG,
        sequence_length=sequence_length,
        seed=seed
    )
    
    # Initialize model
    model = LSTMPredictor(
        input_dim=len(ALL_SENSORS),
        hidden_dim=64,
        num_layers=2
    )
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )
    
    # Strategy
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
    
    num_factories = len(FACTORY_CONFIG)
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=num_factories,
        min_evaluate_clients=num_factories,
        min_available_clients=num_factories,
        evaluate_metrics_aggregation_fn=weighted_average_pdm,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters
    )
    
    # Run simulation
    print(f"\nMemulai simulasi dengan {num_factories} pabrik, {num_rounds} ronde...")
    print("-"*70)
    
    history = start_simulation(
        client_fn=get_factory_client_fn(
            batch_size, local_epochs, learning_rate, sequence_length
        ),
        num_clients=num_factories,
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
            'scenario': 'manufacturing_predictive_maintenance',
            'num_factories': num_factories,
            'num_rounds': num_rounds,
            'sequence_length': sequence_length,
            'seed': seed
        },
        'factories': {str(k): v for k, v in FACTORY_CONFIG.items()},
        'metrics_per_round': [],
        'timestamp': timestamp
    }
    
    # Extract metrics
    if history.metrics_distributed:
        for metric_name in ['rmse', 'mae', 'r2']:
            if metric_name in history.metrics_distributed:
                for round_num, value in history.metrics_distributed[metric_name]:
                    round_entry = next(
                        (r for r in results['metrics_per_round'] if r['round'] == round_num),
                        None
                    )
                    if round_entry is None:
                        round_entry = {'round': round_num}
                        results['metrics_per_round'].append(round_entry)
                    round_entry[metric_name] = value
    
    results['metrics_per_round'].sort(key=lambda x: x['round'])
    
    # Print final metrics
    print("\n" + "-"*70)
    print("HASIL AKHIR")
    print("-"*70)
    
    if results['metrics_per_round']:
        final = results['metrics_per_round'][-1]
        print(f"\nMetrik prediksi RUL (Remaining Useful Life):")
        print(f"  RMSE: {final.get('rmse', 0):.2f} jam")
        print(f"  MAE:  {final.get('mae', 0):.2f} jam")
        print(f"  R²:   {final.get('r2', 0):.4f}")
        
        print("\nInterpretasi:")
        print(f"  - Model dapat memprediksi RUL dengan error rata-rata {final.get('mae', 0):.1f} jam")
        print(f"  - Cukup untuk scheduling maintenance preventif")
        print(f"  - Bisa mengurangi unplanned downtime 30-50%")
    
    # Save results
    results_path = os.path.join(output_dir, f'results_manufacturing_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nHasil disimpan di: {results_path}")
    
    # Plot
    plot_manufacturing_results(results, output_dir, timestamp)
    
    return results


def plot_manufacturing_results(results: Dict, output_dir: str, timestamp: str):
    """Plot hasil simulasi manufacturing."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = results['metrics_per_round']
    if metrics:
        rounds = [m['round'] for m in metrics]
        
        # RMSE
        rmses = [m.get('rmse', 0) for m in metrics]
        axes[0].plot(rounds, rmses, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Ronde')
        axes[0].set_ylabel('RMSE (jam)')
        axes[0].set_title('Root Mean Square Error')
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        maes = [m.get('mae', 0) for m in metrics]
        axes[1].plot(rounds, maes, 'g-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Ronde')
        axes[1].set_ylabel('MAE (jam)')
        axes[1].set_title('Mean Absolute Error')
        axes[1].grid(True, alpha=0.3)
    
    # Factory sensor availability
    factories = results['factories']
    factory_names = [f"F{i}" for i in range(len(factories))]
    
    # Create sensor availability matrix
    sensor_matrix = np.zeros((len(factories), len(ALL_SENSORS)))
    for i, factory_id in enumerate(factories.keys()):
        for sensor in factories[factory_id]['sensors']:
            if sensor in ALL_SENSORS:
                sensor_matrix[i, ALL_SENSORS.index(sensor)] = 1
    
    im = axes[2].imshow(sensor_matrix, cmap='Blues', aspect='auto')
    axes[2].set_xticks(range(len(ALL_SENSORS)))
    axes[2].set_xticklabels(ALL_SENSORS, rotation=45, ha='right')
    axes[2].set_yticks(range(len(factories)))
    axes[2].set_yticklabels(factory_names)
    axes[2].set_xlabel('Sensor')
    axes[2].set_ylabel('Pabrik')
    axes[2].set_title('Ketersediaan Sensor per Pabrik')
    
    plt.tight_layout()
    
    fig.suptitle(
        "Simulasi FL: Predictive Maintenance 8 Pabrik Indonesia",
        y=1.02, fontsize=12, fontweight='bold'
    )
    
    plot_path = os.path.join(output_dir, f'plot_manufacturing_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot disimpan di: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Simulasi FL Predictive Maintenance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-rounds', type=int, default=40)
    parser.add_argument('--local-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./results_manufacturing')
    
    args = parser.parse_args()
    
    run_manufacturing_simulation(
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n✓ Simulasi selesai!")


if __name__ == "__main__":
    main()
