# 融合线性分支的自适应门控FT-Transformer多输出气动系数预测方法

本仓库为论文《融合线性分支的自适应门控FT-Transformer多输出气动系数预测方法》（Adaptive-Gated FT-Transformer with a Linear Branch for Multi-Output Aerodynamic Coefficient Prediction）（投稿至《四川大学学报（自然科学版）》）的开源代码与数据仓库，包含模型实现、对比基线、消融实验及所用数据集。

## 目录结构

```
.
├── main.py                        # 主程序：模型训练与推理流程
├── compare_MLP.py                 # 对比模型：多层感知机（MLP）
├── compare_RR.py                  # 对比模型：岭回归（Ridge Regression）
├── compare_TabTransformer.py      # 对比模型：TabTransformer
├── compare_Transformer.py         # 对比模型：标准Transformer
├── compare_XGBoost.py             # 对比模型：XGBoost
├── ablation_Fixed_weights.py      # 消融实验：固定权重（去除自适应门控机制）
├── ablation_FTTransformer.py      # 消融实验：单独FT-Transformer分支（去除线性分支）
├── data_2822.csv                  # 数据集：RAE2822气动系数样本
├── data_CHN_F1.csv                # 数据集：CHN-F1翼型气动系数样本
└── data_nasa.csv                  # 数据集：NASA CRM_WB气动系数样本
```

## 项目简介

气动系数（如升力系数、阻力系数、力矩系数等）的多输出预测是气动设计与优化中的核心问题之一。本工作提出了一种 **自适应门控 FT-Transformer 与线性分支相结合** 的混合模型：

- **FT-Transformer 分支**：利用特征分词（Feature Tokenizer）与自注意力机制，捕捉特征间的高阶非线性交互关系；
- **线性分支**：保留输入特征与输出之间的低阶线性映射关系，提升模型在训练数据有限或特征线性相关性较强场景下的稳健性；
- **自适应门控机制**：根据输入样本自适应地融合两个分支的输出。

## 环境依赖与安装

本仓库所需的第三方库已列于 `requirements.txt`，请在运行代码前先完成安装：

```bash
pip install -r requirements.txt
```

## 运行说明

**请在运行程序时，将各 `.py` 脚本与所使用的数据集 `.csv` 文件放置于同一文件夹下**，否则程序可能因找不到数据文件而报错。例如：

```
.
├── main.py
├── data_2822.csv
├── data_CHN_F1.csv
└── data_nasa.csv
```

## 数据集说明与引用

本仓库提供的三个数据集（`data_2822.csv`、`data_CHN_F1.csv`、`data_nasa.csv`）均为**处理后的数据**，仅保留了用于构建代理模型（Surrogate Model）所需的输入特征与输出气动系数字段，原始数据来源及引用信息如下：

1. 中国空气动力研究与发展中心计算空气动力研究所. CFD验证与确认数据库[DS/OL]. (2023-01). https://doi.org/10.12176/99.70.00007-V01.

   （Computational Aerodynamics Research Institute, China Aerodynamics Research and Development Center. CFD validation and confirmation database[DS/OL]. (2023-01). https://doi.org/10.12176/99.70.00007-V01.）

2. WHITE P. Dataset for Airbus, Pressure Measurements on the Transonic Aerofoil RAE2822: Version 3[DS/OL]. Loughborough University, 2026[2026-07-21]. DOI: 10.17028/rd.lboro.29383292.

3. NATIONAL AERONAUTICS AND SPACE ADMINISTRATION. NASA Common Research Model: NTF Test 197 Run Data[DS/OL]. [2026-07-21]. https://commonresearchmodel.larc.nasa.gov/experimental-data/ntf-experimental-results/test-197-run-data/

如使用本仓库提供的数据集开展后续研究，请一并引用上述原始数据来源。


## 许可协议

本仓库遵循 MIT License 开源协议，详见 `LICENSE` 文件。



