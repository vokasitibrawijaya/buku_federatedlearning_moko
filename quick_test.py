"""
Quick test script to verify all imports and basic functionality work.
Run this first to ensure the simulation environment is properly set up.
"""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("TESTING FL SIMULATION ENVIRONMENT")
    print("=" * 60)
    
    # Core dependencies
    print("\n[1] Testing core dependencies...")
    try:
        import torch
        print(f"    ✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"    ✗ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"    ✓ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"    ✗ TorchVision: {e}")
        return False
    
    try:
        import flwr
        print(f"    ✓ Flower: {flwr.__version__}")
    except ImportError as e:
        print(f"    ✗ Flower: {e}")
        return False
    
    try:
        import numpy as np
        print(f"    ✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"    ✗ NumPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"    ✓ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"    ✗ Matplotlib: {e}")
        return False
    
    return True

def test_models():
    """Test model definitions."""
    print("\n[2] Testing model definitions...")
    
    try:
        from basic.model import SimpleCNN, CIFAR10CNN, SimpleMLP, get_parameters, set_parameters
        print("    ✓ Imports successful")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    
    import torch
    
    # Test SimpleCNN
    try:
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        print(f"    ✓ SimpleCNN: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"    ✗ SimpleCNN: {e}")
        return False
    
    # Test CIFAR10CNN
    try:
        model = CIFAR10CNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        print(f"    ✓ CIFAR10CNN: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"    ✗ CIFAR10CNN: {e}")
        return False
    
    # Test SimpleMLP
    try:
        model = SimpleMLP(input_dim=30, hidden_dims=[64, 32], num_classes=2)
        x = torch.randn(2, 30)
        output = model(x)
        assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
        print(f"    ✓ SimpleMLP: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"    ✗ SimpleMLP: {e}")
        return False
    
    # Test get/set parameters
    try:
        model1 = SimpleCNN(num_classes=10)
        params = get_parameters(model1)
        model2 = SimpleCNN(num_classes=10)
        set_parameters(model2, params)
        print(f"    ✓ get_parameters/set_parameters: {len(params)} arrays")
    except Exception as e:
        print(f"    ✗ get_parameters/set_parameters: {e}")
        return False
    
    return True

def test_data():
    """Test data loading and partitioning."""
    print("\n[3] Testing data loading and partitioning...")
    
    try:
        from basic.data import load_dataset, partition_iid, partition_dirichlet, get_partition_stats
        print("    ✓ Imports successful")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    
    # Test MNIST loading
    try:
        trainset, testset = load_dataset('mnist', './data')
        print(f"    ✓ MNIST loaded: {len(trainset)} train, {len(testset)} test samples")
    except Exception as e:
        print(f"    ✗ MNIST loading: {e}")
        return False
    
    # Test IID partitioning
    try:
        indices = partition_iid(trainset, num_clients=5, seed=42)
        assert len(indices) == 5, f"Expected 5 clients, got {len(indices)}"
        total_samples = sum(len(idx) for idx in indices)
        print(f"    ✓ IID partitioning: {len(indices)} clients, {total_samples} total samples")
    except Exception as e:
        print(f"    ✗ IID partitioning: {e}")
        return False
    
    # Test Dirichlet partitioning
    try:
        indices = partition_dirichlet(trainset, num_clients=5, alpha=0.5, seed=42)
        assert len(indices) == 5, f"Expected 5 clients, got {len(indices)}"
        print(f"    ✓ Dirichlet partitioning: alpha=0.5, {len(indices)} clients")
    except Exception as e:
        print(f"    ✗ Dirichlet partitioning: {e}")
        return False
    
    # Test partition stats
    try:
        stats = get_partition_stats(trainset, indices)
        print(f"    ✓ Partition stats: {stats['num_clients']} clients, {stats['num_classes']} classes")
    except Exception as e:
        print(f"    ✗ Partition stats: {e}")
        return False
    
    return True

def test_training():
    """Test training functions."""
    print("\n[4] Testing training functions...")
    
    try:
        from basic.train import train, test
        from basic.model import SimpleCNN
        print("    ✓ Imports successful")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SimpleCNN(num_classes=10)
    device = torch.device('cpu')
    
    # Test training
    try:
        history = train(model, dataloader, epochs=1, device=device)
        assert 'loss' in history, "Expected 'loss' in history"
        print(f"    ✓ Training: 1 epoch, loss={history['loss'][-1]:.4f}")
    except Exception as e:
        print(f"    ✗ Training: {e}")
        return False
    
    # Test evaluation
    try:
        loss, accuracy = test(model, dataloader, device)
        print(f"    ✓ Evaluation: loss={loss:.4f}, accuracy={accuracy:.2%}")
    except Exception as e:
        print(f"    ✗ Evaluation: {e}")
        return False
    
    return True

def test_flower_client():
    """Test Flower client implementation."""
    print("\n[5] Testing Flower client...")
    
    try:
        from basic.client import FlowerClient
        from basic.model import SimpleCNN
        from basic.data import load_dataset, partition_iid
        print("    ✓ Imports successful")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    
    import torch
    from torch.utils.data import DataLoader, Subset
    
    # Create client with real data
    try:
        trainset, testset = load_dataset('mnist', './data')
        indices = partition_iid(trainset, num_clients=3, seed=42)
        
        client_trainset = Subset(trainset, indices[0])
        client_testset = testset
        
        trainloader = DataLoader(client_trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(client_testset, batch_size=32)
        
        model = SimpleCNN(num_classes=10)
        device = torch.device('cpu')
        
        client = FlowerClient(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            local_epochs=1
        )
        
        # Test get_parameters
        params = client.get_parameters(config={})
        print(f"    ✓ get_parameters: {len(params)} parameter arrays")
        
        # Test fit (without running full fit for speed)
        print(f"    ✓ FlowerClient created successfully")
    except Exception as e:
        print(f"    ✗ FlowerClient: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_simulation():
    """Test FL simulation (quick version)."""
    print("\n[6] Testing FL simulation components...")
    
    try:
        from basic.model import SimpleCNN, get_parameters, set_parameters
        from basic.data import load_dataset, partition_iid
        from basic.train import train, test
        print("    ✓ All components loaded")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return False
    
    import torch
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    # Setup
    num_clients = 3
    num_rounds = 2
    
    trainset, testset = load_dataset('mnist', './data')
    indices = partition_iid(trainset, num_clients=num_clients, seed=42)
    
    device = torch.device('cpu')
    
    # Global model
    global_model = SimpleCNN(num_classes=10)
    
    # Simulate FL rounds
    try:
        for round_num in range(num_rounds):
            client_params = []
            client_weights = []
            
            global_params = get_parameters(global_model)
            
            # Train each client
            for client_id in range(num_clients):
                local_model = SimpleCNN(num_classes=10)
                set_parameters(local_model, global_params)
                
                client_data = Subset(trainset, indices[client_id][:100])  # Use subset for speed
                trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
                
                train(local_model, trainloader, epochs=1, device=device)
                
                client_params.append(get_parameters(local_model))
                client_weights.append(len(indices[client_id]))
            
            # FedAvg aggregation
            total_weight = sum(client_weights)
            new_params = []
            for param_idx in range(len(client_params[0])):
                weighted_sum = sum(
                    w * params[param_idx] 
                    for w, params in zip(client_weights, client_params)
                )
                new_params.append(weighted_sum / total_weight)
            
            set_parameters(global_model, new_params)
            
            # Evaluate
            testloader = DataLoader(testset, batch_size=64)
            loss, accuracy = test(global_model, testloader, device)
            print(f"    ✓ Round {round_num + 1}/{num_rounds}: loss={loss:.4f}, accuracy={accuracy:.2%}")
    except Exception as e:
        print(f"    ✗ Simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    results = []
    
    results.append(("Core Dependencies", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Model Definitions", test_models()))
        results.append(("Data Loading", test_data()))
        results.append(("Training Functions", test_training()))
        results.append(("Flower Client", test_flower_client()))
        results.append(("FL Simulation", test_simulation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED! The simulation is ready.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
