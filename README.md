# Adaptive-Gated FT-Transformer with a Linear Branch for Multi-Output Aerodynamic Coefficient Prediction

本仓库为论文《Adaptive-Gated FT-Transformer with a Linear Branch for Multi-Output Aerodynamic Coefficient Prediction》（投稿至《四川大学学报》）的开源代码与数据仓库，包含模型实现、对比基线、消融实验及所用数据集。

## 目录结构

```
.
├── main.py                        # 主入口：模型训练与推理流程
├── ablation_Fixed_weights.py      # 消融实验：固定权重（去除自适应门控机制）
├── ablation_FTTransformer.py      # 消融实验：单独 FT-Transformer 分支（去除线性分支）
├── compare_MLP.py                 # 对比基线：多层感知机（MLP）
├── compare_RR.py                  # 对比基线：岭回归（Ridge Regression）
├── compare_TabTransformer.py      # 对比基线：TabTransformer
├── compare_Transformer.py         # 对比基线：标准 Transformer
├── compare_XGBoost.py             # 对比基线：XGBoost
├── data_2822.csv                  # 数据集：气动系数样本（2822 条）
├── data_CHN_F1.csv                # 数据集：CHN-F1 翼型/构型气动数据
└── data_nasa.csv                  # 数据集：NASA 气动数据
```

## 项目简介

气动系数（如升力系数、阻力系数、力矩系数等）的多输出预测是气动设计与优化中的核心问题之一。本工作提出了一种 **自适应门控 FT-Transformer 与线性分支相结合** 的混合模型：

- **FT-Transformer 分支**：利用特征分词（Feature Tokenizer）与自注意力机制，捕捉特征间的高阶非线性交互关系；
- **线性分支**：保留输入特征与输出之间的低阶线性映射关系，提升模型在训练数据有限或特征线性相关性较强场景下的稳健性；
- **自适应门控机制**：根据输入样本自适应地融合两个分支的输出，公式可概括为

\[
\hat{y} = g(x) \odot f_{\text{FTT}}(x) + \big(1 - g(x)\big) \odot f_{\text{Linear}}(x)
\]

其中 \(g(x) \in (0,1)\) 为门控网络输出的权重，用于动态平衡非线性分支与线性分支对最终预测 \(\hat{y}\) 的贡献。

## 文件说明

### 核心代码

| 文件 | 说明 |
|---|---|
| `main.py` | 模型（Adaptive-Gated FT-Transformer + Linear Branch）的训练、验证与测试主流程 |

### 消融实验（Ablation）

| 文件 | 说明 |
|---|---|
| `ablation_Fixed_weights.py` | 将自适应门控替换为固定融合权重，验证门控机制的有效性 |
| `ablation_FTTransformer.py` | 仅保留 FT-Transformer 分支，验证线性分支的必要性 |

### 对比基线（Baselines）

| 文件 | 对应方法 |
|---|---|
| `compare_MLP.py` | 多层感知机 |
| `compare_RR.py` | 岭回归 |
| `compare_TabTransformer.py` | TabTransformer |
| `compare_Transformer.py` | 标准 Transformer |
| `compare_XGBoost.py` | XGBoost |

### 数据集

| 文件 | 说明 |
|---|---|
| `data_2822.csv` | 主实验数据集，共 2822 个样本 |
| `data_CHN_F1.csv` | CHN-F1 构型气动数据，用于跨数据集泛化性验证 |
| `data_nasa.csv` | NASA 公开气动数据，用于跨数据集泛化性验证 |




## 许可协议

本仓库遵循 MIT License 开源协议，详见 `LICENSE` 文件。

