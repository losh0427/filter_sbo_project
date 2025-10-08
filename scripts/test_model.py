import torch
from models import GPSurrogateModel

def true_function(x1, x2):
    return -(x1 - 5)**2 - 2*(x2 - 3)**2 + 25

def botorch_gp_prediction():
    """純BoTorch GP預測"""
    print("=== 純BoTorch GP預測 ===")
    
    # 相同的數據設定
    training_points = [(2, 2), (8, 2), (5, 6), (3, 8)]
    training_y = [14, 14, 7, -29]
    test_point = (6, 4)
    test_y_true = 22
    
    print(f"訓練數據: {training_points} -> {training_y}")
    print(f"測試點: {test_point}, 真實值: {test_y_true}")
    
    # 轉換為張量
    X_train = torch.tensor(training_points, dtype=torch.float64)
    y_train = torch.tensor(training_y, dtype=torch.float64).unsqueeze(-1)
    X_test = torch.tensor([test_point], dtype=torch.float64)
    
    # BoTorch GP模型訓練
    print(f"\n=== BoTorch自動訓練 ===")
    gp_model = GPSurrogateModel()
    gp_model.fit(X_train, y_train)
    
    # 提取學到的超參數
    print(f"\nBoTorch學到的超參數:")
    model = gp_model.get_model()
    covar_module = model.covar_module
    
    lengthscale_tensor = covar_module.base_kernel.lengthscale.squeeze()
    outputscale = covar_module.outputscale.item()
    noise = model.likelihood.noise.item()
    
    if lengthscale_tensor.numel() == 1:
        lengthscales = [lengthscale_tensor.item()]
        print(f"  Lengthscale (isotropic): {lengthscales[0]:.6f}")
    else:
        lengthscales = lengthscale_tensor.tolist()
        print(f"  Lengthscales (anisotropic): {lengthscales}")
        print(f"    維度0 (x1): {lengthscales[0]:.6f}")
        print(f"    維度1 (x2): {lengthscales[1]:.6f}")
    
    print(f"  Outputscale: {outputscale:.6f}")
    print(f"  Noise: {noise:.6f}")
    
    # BoTorch預測
    print(f"\n=== BoTorch預測結果 ===")
    pred_mean, pred_std = gp_model.predict(X_test)
    
    print(f"預測均值: {pred_mean.item():.6f}")
    print(f"預測標準差: {pred_std.item():.6f}")
    print(f"95%信心區間: [{pred_mean.item() - 1.96*pred_std.item():.6f}, {pred_mean.item() + 1.96*pred_std.item():.6f}]")
    print(f"真實值: {test_y_true}")
    print(f"預測誤差: {abs(pred_mean.item() - test_y_true):.6f}")
    
    # 檢查信心區間
    ci_lower = pred_mean.item() - 1.96*pred_std.item()
    ci_upper = pred_mean.item() + 1.96*pred_std.item()
    in_ci = ci_lower <= test_y_true <= ci_upper
    print(f"真實值在95%信心區間內: {'✓' if in_ci else '✗'}")
    
    return pred_mean.item(), pred_std.item(), lengthscales, outputscale, noise

if __name__ == "__main__":
    botorch_gp_prediction()