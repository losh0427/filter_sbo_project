#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cavity Filter Optimization - Package Functionality Test Script
Test all required packages for the cavity filter optimization project
"""

import sys
import traceback

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_basic_imports():
    """Test basic package imports"""
    print_section("BASIC PACKAGE IMPORT TEST")
    
    packages = [
        ("torch", "PyTorch"),
        ("gpytorch", "GPyTorch"), 
        ("botorch", "BoTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas")
    ]
    
    results = {}
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:12} - Import successful")
            results[package] = True
        except ImportError as e:
            print(f"❌ {name:12} - Import failed: {e}")
            results[package] = False
        except Exception as e:
            print(f"⚠️  {name:12} - Unexpected error: {e}")
            results[package] = False
    
    return results

def test_pytorch_functionality():
    """Test PyTorch core functionality"""
    print_section("PYTORCH FUNCTIONALITY TEST")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Test tensor creation and operations
        print("\n📝 Testing tensor operations...")
        x = torch.randn(3, 4, dtype=torch.float64)
        y = torch.randn(4, 2, dtype=torch.float64)
        z = torch.matmul(x, y)
        print(f"   Tensor multiplication: {x.shape} @ {y.shape} = {z.shape}")
        
        # Test default dtype (important for GP models)
        print(f"   Default dtype: {torch.get_default_dtype()}")
        torch.set_default_dtype(torch.float64)
        print(f"   Set to float64: {torch.get_default_dtype()}")
        
        # Test gradient computation
        print("\n📝 Testing gradient computation...")
        x = torch.randn(2, 2, requires_grad=True, dtype=torch.float64)
        y = x.pow(2).sum()
        y.backward()
        print(f"   Gradient shape: {x.grad.shape}")
        print("✅ PyTorch functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_gpytorch_functionality():
    """Test GPyTorch functionality"""
    print_section("GPYTORCH FUNCTIONALITY TEST")
    
    try:
        import torch
        import gpytorch
        from gpytorch.kernels import RBFKernel, MaternKernel
        from gpytorch.means import ConstantMean
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.mlls import ExactMarginalLogLikelihood
        
        print(f"GPyTorch version: {gpytorch.__version__}")
        
        # Create simple GP model
        print("\n📝 Testing GP model creation...")
        
        class SimpleGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(SimpleGP, self).__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = RBFKernel()
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)
        
        # Generate test data
        train_x = torch.randn(10, 2, dtype=torch.float64)
        train_y = torch.randn(10, dtype=torch.float64)
        
        # Create model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SimpleGP(train_x, train_y, likelihood)
        
        print(f"   Model created with kernel: {type(model.covar_module).__name__}")
        print(f"   Training data shape: X={train_x.shape}, y={train_y.shape}")
        
        # Test prediction
        print("\n📝 Testing GP prediction...")
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            test_x = torch.randn(5, 2, dtype=torch.float64)
            predictions = model(test_x)
            mean = predictions.mean
            variance = predictions.variance
            
        print(f"   Prediction mean shape: {mean.shape}")
        print(f"   Prediction variance shape: {variance.shape}")
        print("✅ GPyTorch functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_botorch_functionality():
    """Test BoTorch functionality"""
    print_section("BOTORCH FUNCTIONALITY TEST")
    
    try:
        import torch
        import gpytorch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        try:
            from botorch.fit import fit_gpytorch_mll as fit_model
        except ImportError:
            from botorch.fit import fit_gpytorch_model as fit_model
        from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        
        print("BoTorch components imported successfully")
        
        # Generate test data
        print("\n📝 Testing BoTorch GP model...")
        X = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, 1, dtype=torch.float64)
        
        # Create SingleTaskGP model
        model = SingleTaskGP(
            X, y,
            input_transform=Normalize(d=X.shape[-1]),
            outcome_transform=Standardize(m=y.shape[-1])
        )
        
        # Fit model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_model(mll)
        print(f"   Model fitted with data: X={X.shape}, y={y.shape}")
        
        # Test acquisition function
        print("\n📝 Testing acquisition functions...")
        model.eval()
        
        # Expected Improvement
        best_f = y.max()
        ei = ExpectedImprovement(model=model, best_f=best_f)
        
        # Upper Confidence Bound
        ucb = UpperConfidenceBound(model=model, beta=2.0)
        
        # Test acquisition optimization
        bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        candidate, acq_value = optimize_acqf(
            ei, bounds=bounds, q=1, num_restarts=3, raw_samples=10
        )
        
        print(f"   EI acquisition function created")
        print(f"   UCB acquisition function created")
        print(f"   Optimization result: candidate={candidate.shape}, value={acq_value.item():.4f}")
        print("✅ BoTorch functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ BoTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_scientific_computing():
    """Test NumPy and SciPy functionality"""
    print_section("SCIENTIFIC COMPUTING TEST")
    
    try:
        import numpy as np
        import scipy
        from scipy.optimize import minimize
        from scipy.stats import norm
        
        print(f"NumPy version: {np.__version__}")
        print(f"SciPy version: {scipy.__version__}")
        
        # Test NumPy operations
        print("\n📝 Testing NumPy operations...")
        arr = np.random.randn(100, 10)
        mean_val = np.mean(arr, axis=0)
        std_val = np.std(arr, axis=0)
        print(f"   Array operations: shape={arr.shape}, mean={mean_val.shape}")
        
        # Test SciPy optimization
        print("\n📝 Testing SciPy optimization...")
        def objective(x):
            return x[0]**2 + x[1]**2
        
        result = minimize(objective, x0=[1.0, 1.0], method='L-BFGS-B')
        print(f"   Optimization result: success={result.success}, x={result.x}")
        
        # Test statistical functions
        print("\n📝 Testing statistical functions...")
        sample = norm.rvs(size=100)
        print(f"   Normal distribution sample: mean={np.mean(sample):.3f}")
        
        print("✅ Scientific computing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Scientific computing test failed: {e}")
        traceback.print_exc()
        return False

def test_project_simulation():
    """Test project-specific simulation"""
    print_section("PROJECT SIMULATION TEST")
    
    try:
        import torch
        import numpy as np
        from botorch.models import SingleTaskGP
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        try:
            from botorch.fit import fit_gpytorch_mll as fit_model
        except ImportError:
            from botorch.fit import fit_gpytorch_model as fit_model
        from gpytorch.mlls import ExactMarginalLogLikelihood
        
        print("📝 Simulating cavity filter optimization workflow...")
        
        # Simulate 10-dimensional parameter space (cavity filter parameters)
        bounds = torch.tensor([
            [100.0, 200.0],   # top_width
            [200.0, 400.0],   # bottom_width
            [50.0, 70.0],     # height
            [300.0, 400.0],   # length
            [0.1, 19.9],      # tooth_depth
            [0.05, 50.0],     # tooth_length
            [0.05, 50.0],     # tooth_width
            [0.05, 100.0],    # groove_length
            [0.05, 100.0],    # groove_width
            [1.0, 2.0]        # extension_factor
        ], dtype=torch.float64)
        
        print(f"   Parameter bounds: {bounds.shape} (10-dimensional parameter space)")
        
        # Generate initial training data
        n_init = 20
        X_init = torch.rand(n_init, 10, dtype=torch.float64)
        for i in range(10):
            X_init[:, i] = bounds[i, 0] + X_init[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Simulate objective function (minimize electric field)
        def simulate_objective(X):
            # Simple simulation: quadratic function with noise
            return -(X**2).sum(dim=-1, keepdim=True) + 0.1 * torch.randn(X.shape[0], 1)
        
        y_init = simulate_objective(X_init)
        
        print(f"   Initial training data: X={X_init.shape}, y={y_init.shape}")
        
        # Build GP model
        print("\n📝 Building GP surrogate model...")
        model = SingleTaskGP(X_init, y_init)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_model(mll)
        
        # Acquisition function optimization
        print("\n📝 Optimizing acquisition function...")
        model.eval()
        best_f = y_init.max()
        ei = ExpectedImprovement(model=model, best_f=best_f)
        
        candidate, acq_value = optimize_acqf(
            ei, bounds=bounds.t(), q=1, num_restarts=5, raw_samples=20
        )
        
        print(f"   Next candidate: {candidate.shape}")
        print(f"   Acquisition value: {acq_value.item():.6f}")
        print(f"   Current best objective: {best_f.item():.6f}")
        
        # Simulate one optimization step
        y_new = simulate_objective(candidate)
        print(f"   New objective value: {y_new.item():.6f}")
        
        if y_new.item() > best_f.item():
            print("   🎉 Found improvement!")
        else:
            print("   📊 No improvement (normal in optimization)")
        
        print("✅ Project simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Project simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_data_handling():
    """Test data handling functionality"""
    print_section("DATA HANDLING TEST")
    
    try:
        import torch
        import numpy as np
        
        print("📝 Testing parameter bounds and validation...")
        
        # Parameter bounds (from your project)
        bounds = torch.tensor([
            [100.0, 200.0], [200.0, 400.0], [50.0, 70.0], [300.0, 400.0], [0.1, 19.9],
            [0.05, 50.0], [0.05, 50.0], [0.05, 100.0], [0.05, 100.0], [1.0, 2.0]
        ], dtype=torch.float64)
        
        # Test parameter validation
        test_params = torch.tensor([150.0, 300.0, 60.0, 350.0, 10.0, 25.0, 25.0, 50.0, 50.0, 1.5])
        
        # Check bounds
        valid = True
        for i, (val, bound) in enumerate(zip(test_params, bounds)):
            if val < bound[0] or val > bound[1]:
                valid = False
                break
        
        print(f"   Parameter validation: {'✅ Valid' if valid else '❌ Invalid'}")
        
        # Test parameter clipping
        out_of_bounds = torch.tensor([50.0, 500.0, 30.0, 450.0, 25.0, 100.0, 100.0, 150.0, 150.0, 3.0])
        clipped = torch.zeros_like(out_of_bounds)
        for i, (val, bound) in enumerate(zip(out_of_bounds, bounds)):
            clipped[i] = torch.clamp(val, bound[0], bound[1])
        
        print(f"   Parameter clipping: Original range [{out_of_bounds.min():.1f}, {out_of_bounds.max():.1f}]")
        print(f"                      Clipped range  [{clipped.min():.1f}, {clipped.max():.1f}]")
        
        # Test file I/O simulation
        print("\n📝 Testing data format simulation...")
        
        # Simulate dataset format: 10 parameters + 1 objective value
        n_samples = 5
        param_data = torch.rand(n_samples, 10, dtype=torch.float64)
        objective_data = torch.rand(n_samples, dtype=torch.float64)
        
        # Simulate dataset lines
        dataset_lines = []
        for i in range(n_samples):
            line_data = param_data[i].tolist() + [objective_data[i].item()]
            line_str = ' '.join([f"{val:.6f}" for val in line_data])
            dataset_lines.append(line_str)
        
        print(f"   Generated {len(dataset_lines)} dataset entries")
        print(f"   Example entry: {dataset_lines[0][:50]}...")
        
        print("✅ Data handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print_section("CAVITY FILTER OPTIMIZATION - PACKAGE TEST")
    print("Testing all required packages for the optimization system...")
    print(f"Python version: {sys.version}")
    
    # Run all tests
    test_results = {}
    
    test_results['imports'] = test_basic_imports()
    test_results['pytorch'] = test_pytorch_functionality()
    test_results['gpytorch'] = test_gpytorch_functionality()
    test_results['botorch'] = test_botorch_functionality()
    test_results['scientific'] = test_scientific_computing()
    test_results['simulation'] = test_project_simulation()
    test_results['data'] = test_data_handling()
    
    # Summary
    print_section("TEST SUMMARY")
    
    all_passed = True
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # Import test results
            passed = all(result.values())
            failed_packages = [pkg for pkg, success in result.items() if not success]
        else:
            passed = result
            failed_packages = []
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:10} - {test_name.upper()} test")
        
        if failed_packages:
            print(f"           Failed packages: {', '.join(failed_packages)}")
        
        all_passed = all_passed and passed
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your environment is ready for the project!")
        print("You can now proceed with the cavity filter optimization system.")
    else:
        print("⚠️  SOME TESTS FAILED! Please check the error messages above.")
        print("You may need to reinstall some packages or check your environment.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    main()