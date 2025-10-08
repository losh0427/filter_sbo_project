import torch
import numpy as np
from models import GPSurrogateModel

def true_function(x1, x2):
    """目標函數"""
    return -(x1 - 5)**2 - 2*(x2 - 3)**2 + 25

def extract_botorch_parameters():
    """訓練BoTorch模型並提取超參數"""
    print("=== 第1步：訓練BoTorch模型並提取參數 ===")
    
    # 準備數據
    training_points = [(2, 2), (8, 2), (5, 6), (3, 8)]
    training_y = [14, 14, 7, -29]
    test_point = (6, 4)
    
    X_train = torch.tensor(training_points, dtype=torch.float64)
    y_train = torch.tensor(training_y, dtype=torch.float64).unsqueeze(-1)
    
    # 訓練BoTorch模型
    gp_model = GPSurrogateModel()
    gp_model.fit(X_train, y_train)
    
    # 詳細檢查模型結構
    model = gp_model.get_model()
    print(f"模型類型: {type(model)}")
    print(f"協方差模組: {type(model.covar_module)}")
    print(f"協方差模組結構: {model.covar_module}")
    
    # 檢查是否有input/output transforms
    if hasattr(model, 'input_transform'):
        print(f"Input transform: {model.input_transform}")
    if hasattr(model, 'outcome_transform'):
        print(f"Output transform: {model.outcome_transform}")
    
    covar_module = model.covar_module
    
    # 提取參數
    lengthscale_tensor = covar_module.base_kernel.lengthscale.squeeze()
    outputscale = covar_module.outputscale.item()
    noise = model.likelihood.noise.item()
    
    # 處理各向異性lengthscale
    if lengthscale_tensor.numel() == 1:
        lengthscales = [lengthscale_tensor.item(), lengthscale_tensor.item()]
    else:
        lengthscales = lengthscale_tensor.tolist()
    
    print(f"提取的BoTorch參數:")
    print(f"  Lengthscales: {lengthscales}")
    print(f"  Outputscale: {outputscale}")
    print(f"  Noise: {noise}")
    
    # 測試BoTorch的核函數計算
    print(f"\n測試BoTorch核函數計算:")
    with torch.no_grad():
        X_train_tensor = torch.tensor(training_points, dtype=torch.float64)
        X_test_tensor = torch.tensor([test_point], dtype=torch.float64)
        
        # 如果有transforms，需要應用
        if hasattr(model, 'input_transform') and model.input_transform is not None:
            X_train_transformed = model.input_transform(X_train_tensor)
            X_test_transformed = model.input_transform(X_test_tensor)
            print(f"  原始訓練點: {X_train_tensor}")
            print(f"  轉換後訓練點: {X_train_transformed}")
            print(f"  原始測試點: {X_test_tensor}")
            print(f"  轉換後測試點: {X_test_transformed}")
        else:
            X_train_transformed = X_train_tensor
            X_test_transformed = X_test_tensor
            print(f"  無input transform")
        
        # 計算核矩陣
        K_botorch = model.covar_module(X_train_transformed, X_train_transformed).evaluate()
        k_star_botorch = model.covar_module(X_test_transformed, X_train_transformed).evaluate()
        
        print(f"  BoTorch核矩陣:")
        print(K_botorch.numpy())
        print(f"  BoTorch核向量: {k_star_botorch.squeeze().numpy()}")
    
    return gp_model, lengthscales, outputscale, noise, X_train_transformed, X_test_transformed

def anisotropic_rbf_kernel(x1, x2, lengthscales, outputscale):
    """各向異性RBF核函數"""
    # 確保輸入是torch tensor
    if isinstance(x1, (list, tuple)):
        x1 = torch.tensor(x1, dtype=torch.float64)
    if isinstance(x2, (list, tuple)):
        x2 = torch.tensor(x2, dtype=torch.float64)
    
    # 計算各維度的加權距離平方
    diff = x1 - x2
    weighted_dist_sq = 0.0
    for i in range(len(lengthscales)):
        weighted_dist_sq += (diff[i]**2) / (lengthscales[i]**2)
    
    # RBF核: k(x1,x2) = σ²*exp(-0.5*weighted_dist_sq)
    kernel_value = outputscale * np.exp(-0.5 * weighted_dist_sq)
    
    return kernel_value

def manual_gp_with_botorch_params():
    """使用BoTorch參數進行手算GP預測"""
    
    # 第1步：獲取BoTorch參數
    result = extract_botorch_parameters()
    if len(result) == 6:
        gp_model, lengthscales, outputscale, noise, X_train_transformed, X_test_transformed = result
        use_transforms = True
    else:
        gp_model, lengthscales, outputscale, noise = result
        use_transforms = False
    
    print(f"\n=== 第2步：使用相同參數和數據手算GP ===")
    
    # 數據設定
    training_points = [(2, 2), (8, 2), (5, 6), (3, 8)]
    training_y = [14, 14, 7, -29]
    test_point = (6, 4)
    
    # 如果BoTorch使用了transforms，我們也要使用轉換後的數據
    if use_transforms:
        print("使用BoTorch轉換後的數據進行手算")
        training_points_calc = X_train_transformed.tolist()
        test_point_calc = X_test_transformed[0].tolist()
    else:
        print("使用原始數據進行手算")
        training_points_calc = training_points
        test_point_calc = test_point
    
    print(f"手算使用的訓練點: {training_points_calc}")
    print(f"手算使用的測試點: {test_point_calc}")
    
    n_train = len(training_points_calc)
    
    # 步驟1: 手算核矩陣 K
    print(f"\n步驟1: 使用各向異性RBF計算核矩陣")
    K = np.zeros((n_train, n_train))
    
    for i in range(n_train):
        for j in range(n_train):
            K[i, j] = anisotropic_rbf_kernel(training_points_calc[i], training_points_calc[j], lengthscales, outputscale)
    
    print(f"手算核矩陣 K:")
    print(K)
    
    # 添加噪聲項
    K_noise = K + noise * np.eye(n_train)
    
    # 步驟2: 計算測試點的核向量 k*
    print(f"\n步驟2: 計算核向量 k*")
    k_star = np.zeros(n_train)
    
    for i in range(n_train):
        k_star[i] = anisotropic_rbf_kernel(test_point_calc, training_points_calc[i], lengthscales, outputscale)
    
    print(f"手算核向量 k* = {k_star}")
    
    # 步驟3: 測試點自身的核值 k**
    k_star_star = anisotropic_rbf_kernel(test_point_calc, test_point_calc, lengthscales, outputscale)
    print(f"\n步驟3: 測試點自身核值 k** = {k_star_star:.6f}")
    
    # 步驟4: 求解線性系統
    print(f"\n步驟4: 求解線性系統")
    y_train_array = np.array(training_y)
    
    # 如果BoTorch使用了outcome transform，需要處理y
    if use_transforms and hasattr(gp_model.get_model(), 'outcome_transform') and gp_model.get_model().outcome_transform is not None:
        print("檢測到output transform，但手算中暫時忽略...")
    
    alpha = np.linalg.solve(K_noise, y_train_array)
    print(f"  α = {alpha}")
    
    # 步驟5: 計算後驗均值
    mean = np.dot(k_star, alpha)
    print(f"\n步驟5: 後驗均值 μ* = {mean:.6f}")
    
    # 步驟6: 計算後驗方差
    print(f"\n步驟6: 後驗方差")
    K_inv_k_star = np.linalg.solve(K_noise, k_star)
    var_reduction = np.dot(k_star, K_inv_k_star)
    variance = k_star_star - var_reduction
    std = np.sqrt(max(variance, 1e-10))
    
    print(f"  σ*² = {variance:.6f}")
    print(f"  σ* = {std:.6f}")
    
    print(f"\n=== 手算最終結果 ===")
    print(f"預測均值: {mean:.6f}")
    print(f"預測標準差: {std:.6f}")
    
    # 第3步：BoTorch預測
    print(f"\n=== 第3步：BoTorch預測對比 ===")
    X_test = torch.tensor([test_point], dtype=torch.float64)
    botorch_mean, botorch_std = gp_model.predict(X_test)
    
    print(f"BoTorch結果:")
    print(f"  預測均值: {botorch_mean.item():.6f}")
    print(f"  預測標準差: {botorch_std.item():.6f}")
    
    # 第4步：驗證一致性
    print(f"\n=== 第4步：結果驗證 ===")
    mean_error = abs(mean - botorch_mean.item())
    std_error = abs(std - botorch_std.item())
    
    print(f"均值誤差: {mean_error:.8f}")
    print(f"標準差誤差: {std_error:.8f}")
    
    if mean_error < 1e-5 and std_error < 1e-5:
        print(f"\n✅ 成功！手算與BoTorch結果完全一致")
    else:
        print(f"\n⚠️ 存在差異，可能的原因:")
        print(f"   - Input/Output transforms差異")
        print(f"   - 數值精度問題")
        print(f"   - 矩陣求解方法差異")
        print(f"   - 核函數實現細節差異")
    
    return mean, std, botorch_mean.item(), botorch_std.item()

if __name__ == "__main__":
    manual_gp_with_botorch_params()