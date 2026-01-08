"""
test_simulation.py - Unit Tests untuk Simulasi FL

Memastikan semua komponen simulasi berfungsi dengan benar.

Jalankan dengan: python -m pytest tests/test_simulation.py -v

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.model import SimpleCNN, CIFAR10CNN, SimpleMLP, get_parameters, set_parameters
from basic.data import (
    load_dataset, 
    partition_iid, 
    partition_dirichlet,
    get_partition_stats
)
from basic.train import train, test


class TestModels:
    """Test model definitions."""
    
    def test_simple_cnn_forward(self):
        """Test SimpleCNN forward pass."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_cifar_cnn_forward(self):
        """Test CIFAR10CNN forward pass."""
        model = CIFAR10CNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_simple_mlp_forward(self):
        """Test SimpleMLP forward pass."""
        model = SimpleMLP(input_dim=30, hidden_dims=[64, 32], num_classes=2)
        x = torch.randn(2, 30)
        output = model(x)
        
        assert output.shape == (2, 2)
    
    def test_get_set_parameters(self):
        """Test parameter extraction and setting."""
        model = SimpleCNN(num_classes=10)
        
        # Get parameters
        params = get_parameters(model)
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)
        
        # Create new model and set parameters
        model2 = SimpleCNN(num_classes=10)
        set_parameters(model2, params)
        
        # Check parameters are same
        params2 = get_parameters(model2)
        for p1, p2 in zip(params, params2):
            np.testing.assert_array_almost_equal(p1, p2)


class TestDataPartitioning:
    """Test data partitioning functions."""
    
    @pytest.fixture
    def mnist_dataset(self):
        """Load MNIST dataset for testing."""
        trainset, _ = load_dataset('mnist', './test_data')
        return trainset
    
    def test_partition_iid(self, mnist_dataset):
        """Test IID partitioning."""
        num_clients = 5
        indices = partition_iid(mnist_dataset, num_clients, seed=42)
        
        assert len(indices) == num_clients
        assert all(len(idx) > 0 for idx in indices)
        
        # Check no overlap
        all_indices = []
        for idx in indices:
            all_indices.extend(idx)
        assert len(all_indices) == len(set(all_indices))
    
    def test_partition_dirichlet(self, mnist_dataset):
        """Test Dirichlet (non-IID) partitioning."""
        num_clients = 5
        alpha = 0.5
        indices = partition_dirichlet(mnist_dataset, num_clients, alpha, seed=42)
        
        assert len(indices) == num_clients
        assert all(len(idx) > 0 for idx in indices)
    
    def test_partition_stats(self, mnist_dataset):
        """Test partition statistics."""
        indices = partition_iid(mnist_dataset, num_clients=5, seed=42)
        stats = get_partition_stats(mnist_dataset, indices)
        
        assert 'num_clients' in stats
        assert 'num_classes' in stats
        assert 'samples_per_client' in stats
        assert 'label_distribution' in stats
        
        assert stats['num_clients'] == 5
        assert len(stats['samples_per_client']) == 5


class TestTraining:
    """Test training functions."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup for training tests."""
        model = SimpleCNN(num_classes=10)
        device = torch.device('cpu')
        
        # Create dummy data
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        return model, dataloader, device
    
    def test_train_one_epoch(self, setup_training):
        """Test single epoch training."""
        model, trainloader, device = setup_training
        
        history = train(model, trainloader, epochs=1, device=device)
        
        assert 'loss' in history
        assert len(history['loss']) == 1
        assert history['loss'][0] > 0
    
    def test_evaluate(self, setup_training):
        """Test model evaluation."""
        model, testloader, device = setup_training
        
        loss, accuracy = test(model, testloader, device)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestIntegration:
    """Integration tests for complete FL workflow."""
    
    def test_full_workflow(self):
        """Test complete FL workflow (simplified)."""
        # Create model
        model = SimpleCNN(num_classes=10)
        
        # Create dummy data for 3 clients
        num_clients = 3
        client_data = []
        for _ in range(num_clients):
            x = torch.randn(50, 1, 28, 28)
            y = torch.randint(0, 10, (50,))
            dataset = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16)
            client_data.append(loader)
        
        device = torch.device('cpu')
        
        # Simulate FL rounds
        for round_num in range(2):
            # Get global parameters
            global_params = get_parameters(model)
            
            # Train each client
            client_params = []
            client_weights = []
            
            for client_id, trainloader in enumerate(client_data):
                # Create local model
                local_model = SimpleCNN(num_classes=10)
                set_parameters(local_model, global_params)
                
                # Train
                train(local_model, trainloader, epochs=1, device=device)
                
                # Collect parameters
                client_params.append(get_parameters(local_model))
                client_weights.append(len(trainloader.dataset))
            
            # Simple FedAvg aggregation
            total_weight = sum(client_weights)
            new_params = []
            
            for param_idx in range(len(client_params[0])):
                weighted_sum = sum(
                    w * params[param_idx] 
                    for w, params in zip(client_weights, client_params)
                )
                new_params.append(weighted_sum / total_weight)
            
            # Update global model
            set_parameters(model, new_params)
        
        # Final evaluation
        test_x = torch.randn(20, 1, 28, 28)
        test_y = torch.randint(0, 10, (20,))
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
        
        loss, accuracy = test(model, test_loader, device)
        
        # Just check it runs without errors
        assert loss >= 0
        assert 0 <= accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
