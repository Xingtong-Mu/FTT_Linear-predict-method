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


## 许可协议

本仓库遵循 MIT License 开源协议，详见 `LICENSE` 文件。

