import torch
from models import GPSurrogateModel

def true_function(x1, x2):
    return -(x1 - 5)**2 - 2*(x2 - 3)**2 + 25

def debug_botorch_hyperparams():
    """診斷BoTorch學到的超參數"""
    print("=== BoTorch超參數診斷 ===")
    
    # 準備數據
    training_points = [(2, 2), (8, 2), (5, 6), (3, 8)]
    training_y = [14, 14, 7, -29]
    test_point = (6, 4)
    
    X_train = torch.tensor(training_points, dtype=torch.float64)
    y_train = torch.tensor(training_y, dtype=torch.float64).unsqueeze(-1)
    X_test = torch.tensor([test_point], dtype=torch.float64)
    
    # 訓練GP模型
    gp_model = GPSurrogateModel()
    gp_model.fit(X_train, y_train)
    
    # 提取超參數
    model = gp_model.get_model()
    
    print("BoTorch學到的超參數:")
    try:
        # 獲取covariance module
        covar_module = model.covar_module
        print(f"Covariance module type: {type(covar_module)}")
        
        # 嘗試提取RBF kernel參數
        if hasattr(covar_module, 'base_kernel'):
            lengthscale_tensor = covar_module.base_kernel.lengthscale
            outputscale = covar_module.outputscale.item()
            
            # 檢查lengthscale是標量還是向量
            if lengthscale_tensor.numel() == 1:
                lengthscale = lengthscale_tensor.item()
                print(f"  Lengthscale (scalar): {lengthscale:.6f}")
            else:
                lengthscale_values = lengthscale_tensor.squeeze().tolist()
                print(f"  Lengthscale (vector): {lengthscale_values}")
                print(f"    維度0 (x1): {lengthscale_values[0]:.6f}")
                print(f"    維度1 (x2): {lengthscale_values[1]:.6f}")
                
            print(f"  Outputscale: {outputscale:.6f}")
        else:
            print(f"  Complex kernel structure: {covar_module}")
            
        # 噪聲參數
        noise = model.likelihood.noise.item()
        print(f"  Noise: {noise:.6f}")
        
        # 比較與手算參數
        print(f"\n手算使用的參數:")
        print(f"  Lengthscale: 3.000000 (所有維度相同)")
        print(f"  Outputscale: 10.000000")
        print(f"  Noise: 0.100000")
        
        print(f"\n參數差異分析:")
        if hasattr(covar_module, 'base_kernel'):
            if lengthscale_tensor.numel() == 1:
                lengthscale = lengthscale_tensor.item()
                print(f"  Lengthscale比值: {lengthscale/3.0:.2f}x")
            else:
                lengthscale_values = lengthscale_tensor.squeeze().tolist()
                print(f"  Lengthscale比值 (維度0): {lengthscale_values[0]/3.0:.2f}x")
                print(f"  Lengthscale比值 (維度1): {lengthscale_values[1]/3.0:.2f}x")
                print(f"  ⚠️ BoTorch學到了各向異性長度尺度！")
                
            print(f"  Outputscale比值: {outputscale/10.0:.2f}x")
            print(f"  Noise比值: {noise/0.1:.2f}x")
            
            # 分析異常參數
            if noise > 1.0:
                print(f"  🚨 噪聲過大: {noise:.3f} >> 0.1")
            if outputscale > 100:
                print(f"  🚨 輸出尺度過大: {outputscale:.3f} >> 10.0")
            if lengthscale_tensor.numel() > 1:
                if any(l < 0.1 or l > 50 for l in lengthscale_values):
                    print(f"  🚨 長度尺度異常: {lengthscale_values}")
            elif lengthscale_tensor.numel() == 1 and (lengthscale < 0.1 or lengthscale > 50):
                print(f"  🚨 長度尺度異常: {lengthscale:.3f}")
        
    except Exception as e:
        print(f"提取超參數時出錯: {e}")
        
    # 預測結果
    pred_mean, pred_std = gp_model.predict(X_test)
    print(f"\nBoTorch預測結果:")
    print(f"  均值: {pred_mean.item():.4f}")
    print(f"  標準差: {pred_std.item():.4f}")
    
    return gp_model

def test_with_fixed_hyperparams():
    """測試用固定超參數是否能改善BoTorch結果"""
    print(f"\n=== 測試固定超參數的BoTorch ===")
    
    # 這需要修改models.py來支持固定超參數
    # 目前只是演示概念
    print("注意：需要修改GPSurrogateModel來支持固定超參數")
    print("建議的修改：在__init__中添加固定超參數選項")

def suggest_solutions():
    """建議解決方案"""
    print(f"\n=== 問題解決建議 ===")
    print("🔍 根據診斷結果，BoTorch的問題可能是:")
    print("1. **各向異性長度尺度**: 對2維輸入學到了不同的長度尺度")
    print("   - 可能導致某個維度的影響被放大或縮小")
    print("   - 在小數據集上容易產生不穩定的優化結果")
    
    print("\n2. **超參數優化失敗**: 可能的原因:")
    print("   - 數據量太少(4點)，無法可靠估計多個超參數")
    print("   - 優化陷入局部最優，學到病態參數")
    print("   - 默認初始化和約束不適合這個問題")
    
    print("\n💡 實際解決方案:")
    print("1. **投影片重點調整**:")
    print("   - 強調手算的參數選擇邏輯和合理性")
    print("   - 將BoTorch結果作為「自動優化失敗」的案例")
    print("   - 突出理論指導在小數據集上的重要性")
    
    print("\n2. **工程實踐指導**:")
    print("   - 小數據集優先使用理論指導的固定參數")
    print("   - 大數據集再考慮自動超參數優化")
    print("   - 始終對自動優化結果進行合理性檢查")
    
    print("\n3. **模型壓縮項目啟示**:")
    print("   - Surrogate model構建需要理論基礎")
    print("   - 不能完全依賴自動化工具")
    print("   - 小樣本場景下手工調參可能更可靠")
    
    print("\n🎯 這個發現的教學價值:")
    print("   ✅ 證明了GP理論學習的實用性")
    print("   ✅ 展示了工具使用的注意事項")
    print("   ✅ 提供了真實的工程經驗")
    print("   ✅ 比「完美一致」的結果更有啟發性！")

if __name__ == "__main__":
    gp_model = debug_botorch_hyperparams()
    test_with_fixed_hyperparams()
    suggest_solutions()