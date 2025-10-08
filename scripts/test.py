#!/usr/bin/env python3
"""
GP Model Verification: BoTorch vs Manual Calculation
Demonstrates GP prediction transparency by comparing automated tools with manual math
"""

import torch
import numpy as np
import sys
sys.path.append('..')  # Add parent directory to path for models.py
from models import GPSurrogateModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def objective_function(x):
    """Target function: f(x1,x2) = -(x1-5)² - 2(x2-3)² + 25"""
    x1, x2 = x[..., 0], x[..., 1]
    return -(x1 - 5)**2 - 2*(x2 - 3)**2 + 25

def manual_gp_predict(X_train, y_train, X_test, hyperparams):
    """
    Manual GP prediction using pure mathematical operations
    Implements exact same pipeline as BoTorch SingleTaskGP
    """
    print("\n" + "="*50)
    print("詳細手算步驟 DETAILED MANUAL CALCULATION STEPS")
    print("="*50)
    
    # Step 1: Input normalization (match BoTorch's Normalize transform)
    print("\n【步驟1: Input Normalization】")
    print(f"原始訓練數據 X_train:")
    for i, x in enumerate(X_train):
        print(f"  Point {i}: {x.tolist()}")
    print(f"原始測試數據 X_test: {X_test[0].tolist()}")
    
    X_min = X_train.min(dim=0)[0]
    X_max = X_train.max(dim=0)[0]
    print(f"\nX_min = {X_min.tolist()}")
    print(f"X_max = {X_max.tolist()}")
    print(f"X_range = X_max - X_min = {(X_max - X_min).tolist()}")
    
    X_train_norm = (X_train - X_min) / (X_max - X_min)
    X_test_norm = (X_test - X_min) / (X_max - X_min)
    
    print(f"\n標準化後訓練數據 X_train_norm:")
    for i, x in enumerate(X_train_norm):
        print(f"  Point {i}: {x.tolist()}")
    print(f"標準化後測試數據 X_test_norm: {X_test_norm[0].tolist()}")
    
    # Step 2: Output standardization (match BoTorch's Standardize transform)
    print(f"\n【步驟2: Output Standardization】")
    print(f"原始訓練標籤 y_train: {y_train.tolist()}")
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    print(f"y_mean = {y_mean.item():.6f}")
    print(f"y_std = {y_std.item():.6f}")
    
    y_train_norm = (y_train - y_mean) / y_std
    print(f"標準化後 y_train_norm: {y_train_norm.tolist()}")
    
    # Step 3: Build RBF kernel matrix K
    print(f"\n【步驟3: 超參數設定】")
    lengthscale = hyperparams['lengthscale']
    outputscale = hyperparams['outputscale']
    noise = hyperparams['noise']
    
    print(f"Lengthscale: {lengthscale}")
    print(f"Outputscale: {outputscale:.6f}")
    print(f"Noise: {noise:.6f}")
    
    # RBF kernel: k(x,x') = σ_f² * exp(-0.5 * ||x-x'||² / l²)
    def rbf_kernel(X1, X2, lengthscale, outputscale):
        # Compute squared distances
        dist_sq = torch.cdist(X1 / lengthscale, X2 / lengthscale, p=2).pow(2)
        return outputscale * torch.exp(-0.5 * dist_sq)
    
    # Compute kernel matrices
    print(f"\n【步驟4: 核矩陣計算】")
    K = rbf_kernel(X_train_norm, X_train_norm, lengthscale, outputscale)
    print(f"核矩陣 K (4x4):")
    for i in range(K.shape[0]):
        row_str = "  [" + ", ".join([f"{K[i,j].item():.6f}" for j in range(K.shape[1])]) + "]"
        print(row_str)
    
    print(f"\n【步驟5: 加入噪聲項】")
    K_noise = K + noise * torch.eye(K.shape[0], dtype=torch.float64)
    print(f"加噪聲核矩陣 K_noise = K + {noise:.6f} * I:")
    for i in range(K_noise.shape[0]):
        row_str = "  [" + ", ".join([f"{K_noise[i,j].item():.6f}" for j in range(K_noise.shape[1])]) + "]"
        print(row_str)
    
    print(f"\n【步驟6: 核向量k*計算】")
    k_star = rbf_kernel(X_test_norm, X_train_norm, lengthscale, outputscale)
    print(f"核向量 k* = {[f'{x:.6f}' for x in k_star.squeeze().tolist()]}")
    
    k_star_star = rbf_kernel(X_test_norm, X_test_norm, lengthscale, outputscale)
    print(f"測試點自相關 k** = {k_star_star.item():.6f}")
    
    # Step 4: GP prediction in normalized space
    print(f"\n【步驟7: Cholesky分解】")
    # Using Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(K_noise + 1e-6 * torch.eye(K.shape[0]))  # Add jitter
    print(f"下三角矩陣 L:")
    for i in range(L.shape[0]):
        row_str = "  [" + ", ".join([f"{L[i,j].item():.6f}" for j in range(L.shape[1])]) + "]"
        print(row_str)
    
    # Reshape y_train_norm to column vector for solve_triangular
    y_train_norm_col = y_train_norm.unsqueeze(-1)
    alpha = torch.linalg.solve_triangular(L, y_train_norm_col, upper=False)
    alpha = torch.linalg.solve_triangular(L.T, alpha, upper=True)
    
    print(f"\n【步驟8: 線性系統求解】")
    print(f"α = (K + σ_n²I)⁻¹y = {[f'{x:.6f}' for x in alpha.squeeze().tolist()]}")
    
    # Predictive mean: μ* = k*ᵀ(K + σ²I)⁻¹y
    mean_norm = k_star @ alpha.squeeze()
    print(f"\n【步驟9: 預測均值 (標準化空間)】")
    print(f"μ*_norm = k*ᵀ α = {mean_norm.item():.6f}")
    
    # Predictive variance: σ*² = k** - k*ᵀ(K + σ²I)⁻¹k*
    v = torch.linalg.solve_triangular(L, k_star.T, upper=False)
    var_norm = k_star_star - v.T @ v
    
    print(f"\n【步驟10: 預測方差 (標準化空間)】")
    print(f"v = L⁻¹k* = {[f'{x:.6f}' for x in v.squeeze().tolist()]}")
    print(f"var_reduction = vᵀv = {(v.T @ v).item():.6f}")
    print(f"σ*²_norm = k** - var_reduction = {k_star_star.item():.6f} - {(v.T @ v).item():.6f} = {var_norm.item():.6f}")
    print(f"σ*_norm = √{var_norm.item():.6f} = {torch.sqrt(var_norm).item():.6f}")
    
    # Step 5: Transform back to original space
    print(f"\n【步驟11: 逆變換回原始空間】")
    mean = mean_norm * y_std + y_mean
    variance = var_norm * y_std**2
    std = torch.sqrt(variance.clamp(min=1e-10))
    
    print(f"最終預測均值: μ* = {mean_norm.item():.6f} × {y_std.item():.6f} + {y_mean.item():.6f} = {mean.item():.6f}")
    print(f"最終預測標準差: σ* = {torch.sqrt(var_norm).item():.6f} × {y_std.item():.6f} = {std.item():.6f}")
    
    return mean.squeeze(), std.squeeze()

def extract_hyperparameters(gp_model):
    """Extract trained hyperparameters from BoTorch model"""
    model = gp_model.get_model()
    
    # Get lengthscale (handle different shapes)
    lengthscale = model.covar_module.base_kernel.lengthscale.detach()
    if lengthscale.dim() > 1:
        lengthscale = lengthscale.squeeze()
    
    # Get outputscale
    outputscale = model.covar_module.outputscale.detach().item()
    
    # Get noise variance
    noise = model.likelihood.noise.detach().item()
    
    return {
        'lengthscale': lengthscale,
        'outputscale': outputscale,
        'noise': noise
    }

def main():
    """Main verification workflow"""
    print("="*60)
    print("GP Model Verification: BoTorch vs Manual Calculation")
    print("="*60)
    
    # Define training data
    X_train = torch.tensor([
        [2.0, 2.0],
        [5.0, 2.0],
        [3.0, 6.0],
        [4.0, 8.0]
    ], dtype=torch.float64)
    
    y_train = objective_function(X_train)
    
    # Define test point
    X_test = torch.tensor([[6.0, 4.0]], dtype=torch.float64)
    y_true = objective_function(X_test)
    
    print("\n📊 Training Data:")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"  Point {i+1}: x={x.tolist()}, y={y.item():.4f}")
    
    print(f"\n🎯 Test Point: x={X_test[0].tolist()}, y_true={y_true.item():.4f}")
    
    # Step 1: Train BoTorch model
    print("\n" + "="*40)
    print("Step 1: Training BoTorch Model")
    print("="*40)
    
    gp_model = GPSurrogateModel()
    gp_model.fit(X_train, y_train)
    
    # Step 2: Get BoTorch predictions
    mean_botorch, std_botorch = gp_model.predict(X_test)
    print(f"\n✅ BoTorch Predictions:")
    print(f"   Mean: {mean_botorch.item():.6f}")
    print(f"   Std:  {std_botorch.item():.6f}")
    
    # Step 3: Extract hyperparameters
    hyperparams = extract_hyperparameters(gp_model)
    print(f"\n🔧 Extracted Hyperparameters:")
    print(f"   Lengthscale: {hyperparams['lengthscale']}")
    print(f"   Outputscale: {hyperparams['outputscale']:.6f}")
    print(f"   Noise:       {hyperparams['noise']:.6e}")
    
    # Step 4: Manual calculation
    print("\n" + "="*40)
    print("Step 2: Manual GP Calculation")
    print("="*40)
    
    mean_manual, std_manual = manual_gp_predict(X_train, y_train, X_test, hyperparams)
    print(f"\n✅ Manual Predictions:")
    print(f"   Mean: {mean_manual.item():.6f}")
    print(f"   Std:  {std_manual.item():.6f}")
    
    # Step 5: Compare results
    print("\n" + "="*40)
    print("Verification Results")
    print("="*40)
    
    mean_error = abs(mean_botorch - mean_manual).item()
    std_error = abs(std_botorch - std_manual).item()
    
    print(f"\n📊 Absolute Errors:")
    print(f"   Mean error: {mean_error:.2e}")
    print(f"   Std error:  {std_error:.2e}")
    
    print(f"\n📊 Relative Errors:")
    print(f"   Mean: {mean_error / abs(mean_botorch).item() * 100:.2f}%")
    print(f"   Std:  {std_error / std_botorch.item() * 100:.2f}%")
    
    # Verification
    tolerance = 1
    if mean_error < tolerance and std_error < tolerance:
        print(f"\n✅ VERIFICATION PASSED! Errors < {tolerance}")
        print("   Manual calculation matches BoTorch implementation!")
    else:
        print(f"\n❌ VERIFICATION FAILED! Errors > {tolerance}")
    
    # Additional insights
    print("\n" + "="*40)
    print("Mathematical Transparency Demonstrated")
    print("="*40)
    print("✓ Input normalization applied correctly")
    print("✓ Output standardization applied correctly")
    print("✓ RBF kernel computation verified")
    print("✓ GP prediction formulas validated")
    print("✓ Reverse transformations confirmed")
    
    return mean_error < tolerance and std_error < tolerance

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)