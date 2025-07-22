# 目錄結構預覽

## 專案根目錄結構
```
project_root/
├── scripts/                          # 所有腳本和配置文件
│   ├── functions.py                   # 通用功能模組
│   ├── validation.py                  # 驗證功能模組  
│   ├── initial.py                     # 初始化完整流程
│   ├── hfss_script.py                # 增強版 HFSS 腳本
│   ├── generate_sample_input.py       # 生成樣本工具
│   ├── parameters_config.json         # 參數配置文件
│   └── README.md                      # 使用說明
├── Data1/                            # 輸入輸出數據目錄 (扁平結構)
│   ├── input0.txt                    # 輸入參數文件
│   ├── input1.txt
│   ├── input2.txt
│   ├── ...
│   ├── output0.fld                   # 仿真輸出文件
│   ├── output1.fld
│   ├── output2.fld
│   └── ...
└── Data/                             # 數據集目錄
    └── dataset.txt                   # 最終整合數據集
```

## 使用流程

### 1. 初始化階段 (在 scripts 目錄下執行)
```bash
cd scripts

# 生成300組隨機參數 (創建 ../Data1/input0.txt 到 input299.txt)
python initial.py --n_samples 300

# 或生成單個樣本進行測試
python generate_sample_input.py
```

### 2. HFSS 仿真階段
```bash
# 在 Ansys Electronics Desktop 中執行
python hfss_script.py

# 或在命令行執行 (如果有授權)
cd scripts
python hfss_script.py
```

### 3. 結果整合階段
```bash
cd scripts

# 整合所有結果生成數據集 (創建 ../Data/dataset.txt)
python initial.py --integrate

# 或驗證結果完整性
python initial.py --validate
```

## 路徑說明

### 從 scripts/ 目錄執行時的路徑對應
- `../Data1/input{i}.txt` → 輸入參數文件
- `../Data1/output{i}.fld` → 仿真輸出文件  
- `../Data/dataset.txt` → 最終數據集文件 (新的Data目錄)
- `parameters_config.json` → 參數配置文件 (在同目錄)

### 配置文件自動查找順序
1. 當前目錄
2. scripts/ 目錄  
3. 腳本文件所在目錄

## 自定義路徑

所有腳本都支持自定義路徑參數：

```bash
# 使用自定義數據目錄
python initial.py --data_dir /path/to/custom/data

# 使用自定義數據集文件
python initial.py --dataset_file /path/to/custom/dataset.txt

# 使用自定義配置文件
python initial.py --config_file custom_config.json
```

## 注意事項

1. **執行目錄**: 建議在 scripts/ 目錄下執行所有 Python 腳本
2. **目錄分離**: Data1/ 存放輸入輸出文件，Data/ 存放最終數據集
3. **文件命名**: 嚴格按照 input{i}.txt 和 output{i}.fld 格式命名
4. **權限問題**: 確保對 Data1/ 和 Data/ 目錄有讀寫權限
5. **Ansys 授權**: HFSS 仿真需要有效的 Ansys Electronics Desktop 授權