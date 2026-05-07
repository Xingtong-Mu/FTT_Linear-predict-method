# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge  # 导入岭回归
import warnings
warnings.filterwarnings('ignore')

# ==================== 中文字体设置 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 随机种子 ====================
seed = 42
np.random.seed(seed)

# ==================== 参数定义 ====================
# 岭回归的主要参数是 alpha (即公式中的 lambda)
# alpha 越大，正则化越强，模型越简单
alpha_value = 1.0 

# ==================== 数据加载与预处理 ====================
print("开始加载数据...")

# 确保文件路径正确
df = pd.read_excel('data_all.xlsx')

feature_columns = ['MA', 'BETA', 'AA']
target_columns = ['CL', 'CD', 'CN', 'CA', 'CM']

X = df[feature_columns].values
y = df[target_columns].values

print(f"特征维度: {X.shape}, 目标维度: {y.shape}")

# 划分训练集和测试集 (岭回归不需要单独的验证集用于早停，但为了结构一致可以保留)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 岭回归对特征缩放非常敏感，必须进行标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ==================== 岭回归模型训练 ====================
print(f"开始训练岭回归 (alpha={alpha_value})...")

# 创建岭回归模型
# random_state 保证结果可复现
model = Ridge(alpha=alpha_value, random_state=seed)

# 训练模型
model.fit(X_train_scaled, y_train_scaled)

print("训练完成！")

# ==================== 评估函数 ====================
def evaluate_sklearn_model(model, X_scaled, y_scaled, scaler_y):
    # 预测 (在标准化尺度上)
    y_pred_scaled = model.predict(X_scaled)
    
    # 逆标准化回原始尺度
    y_preds_orig = scaler_y.inverse_transform(y_pred_scaled)
    y_trues_orig = scaler_y.inverse_transform(y_scaled)

    # 计算总体指标
    total_mse = mean_squared_error(y_trues_orig, y_preds_orig)
    total_rmse = np.sqrt(total_mse)
    total_mae = mean_absolute_error(y_trues_orig, y_preds_orig)
    total_r2 = r2_score(y_trues_orig, y_preds_orig)

    # 模拟你之前的 Combined Loss (MSE + 0.33 * MAE)
    total_combined = total_mse + 0.33 * total_mae

    n_outputs = y_trues_orig.shape[1]
    component_metrics = {}

    for i in range(n_outputs):
        y_true_i = y_trues_orig[:, i]
        y_pred_i = y_preds_orig[:, i]

        rmse_i = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
        mae_i = mean_absolute_error(y_true_i, y_pred_i)
        r2_i = r2_score(y_true_i, y_pred_i)

        # 计算平均相对误差 MRE
        epsilon = 1e-8
        mre_i = np.mean(
            np.abs((y_true_i - y_pred_i) / (np.abs(y_true_i) + epsilon))
        )

        component_metrics[i] = {
            'RMSE': rmse_i,
            'MAE': mae_i,
            'R2': r2_i,
            'MRE': mre_i
        }

    overall_metrics = {
        'MSE': total_mse,
        'RMSE': total_rmse,
        'MAE': total_mae,
        'R2': total_r2,
        'Combined_Loss': total_combined
    }

    return overall_metrics, component_metrics, (y_trues_orig, y_preds_orig)

# ==================== 结果评估 ====================

print("\n===== 训练集评估结果 =====")
train_overall, _, _ = evaluate_sklearn_model(model, X_train_scaled, y_train_scaled, scaler_y)
print(f"总体RMSE={train_overall['RMSE']:.4f}, R2={train_overall['R2']:.4f}")

print("\n===== 测试集评估结果（原始尺度） =====")
test_overall, test_component, (test_true, test_pred) = evaluate_sklearn_model(
    model, X_test_scaled, y_test_scaled, scaler_y
)

print(
    f"总体指标: "
    f"MSE={test_overall['MSE']:.4f}, "
    f"RMSE={test_overall['RMSE']:.4f}, "
    f"MAE={test_overall['MAE']:.4f}, "
    f"R2={test_overall['R2']:.4f}, "
    f"Combined={test_overall['Combined_Loss']:.4f}"
)

print("\n各分量详细指标:")
for i, name in enumerate(target_columns):
    m = test_component[i]
    print(
        f"{name}: "
        f"RMSE={m['RMSE']:.4f}, "
        f"MAE={m['MAE']:.4f}, "
        f"R2={m['R2']:.4f}, "
        f"MRE={m['MRE']:.4%}"
    )

# ==================== 可视化 ====================
def plot_true_vs_pred(
    true,
    pred,
    target_names,
    component_metrics,
    save_path='ridge_comparison.png'
):
    n = len(target_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], alpha=0.6, edgecolors='k', s=30)

        min_val = min(true[:, i].min(), pred[:, i].min())
        max_val = max(true[:, i].max(), pred[:, i].max())

        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{name}\n(R² = {component_metrics[i]["R2"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if n < len(axes):
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\n对比图已保存至 {save_path}")

plot_true_vs_pred(
    test_true,
    test_pred,
    target_columns,
    test_component,
    save_path='ridge_comparison.png'
)

# 查看各特征权重（解释性分析）
print("\n===== 岭回归特征权重 (系数) =====")
coeffs = pd.DataFrame(
    scaler_y.inverse_transform(model.coef_).T if hasattr(model, 'coef_') else "N/A", 
    index=feature_columns, 
    columns=target_columns
)
# 注意：由于是在标准化后的空间训练，直接看 model.coef_ 更准确
coeffs_scaled = pd.DataFrame(
    model.coef_, 
    index=target_columns, 
    columns=feature_columns
).T
print("标准化空间下的特征权重（绝对值越大越重要）：")
print(coeffs_scaled)

print("\n岭回归实验完成！")



