# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局字体设置 ====================
# 中文用宋体(SimHei)，英文/数字用Times New Roman
# 设置中文字体
# ==================== 全局字体设置 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(0)

# ==================== 参数定义 ====================
epochs = 300

# ==================== 数据加载与预处理 ====================
print("开始加载数据...")
df = pd.read_excel('data_all.xlsx')
feature_columns = ['MA', 'BETA', 'AA']          # AA 为攻角
target_columns = ['CL', 'CD', 'CN', 'CA', 'CM']
X = df[feature_columns].values
y = df[target_columns].values
print(X.shape, y.shape)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)
print("数据标准化完成")

# 保存数据到Excel（略）
learn_data = pd.DataFrame(np.vstack([X_temp, X_test]), columns=feature_columns)
learn_targets = pd.DataFrame(np.vstack([y_temp, y_test]), columns=target_columns)
dataset_type = ['训练集'] * len(X_train) + ['验证集'] * len(X_val) + ['测试集'] * len(X_test)
learn_data['数据集类型'] = dataset_type
learn_combined = pd.concat([learn_data, learn_targets], axis=1)
with pd.ExcelWriter('学习数据集.xlsx') as writer:
    learn_combined.to_excel(writer, sheet_name='完整数据', index=False)
    learn_combined[learn_combined['数据集类型'] == '训练集'].to_excel(writer, sheet_name='训练集', index=False)
    learn_combined[learn_combined['数据集类型'] == '验证集'].to_excel(writer, sheet_name='验证集', index=False)
    learn_combined[learn_combined['数据集类型'] == '测试集'].to_excel(writer, sheet_name='测试集', index=False)
print("数据已保存至 '学习数据集.xlsx'")

# 转换为Tensor
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==================== 模型定义（含自适应融合 + 返回权重选项） ====================
class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * nn.functional.relu(b)

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.randn(num_features, d_model))
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return x

class TransformerEncoderLayer_ReGLU(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.001):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = ReGLU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class FTTransformer(nn.Module):
    def __init__(self, num_features, num_outputs, d_model=64, nhead=4, num_layers=6,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.num_outputs = num_outputs
        self.numerical_tokenizer = NumericalFeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))
        self.dropout_layer = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer_ReGLU(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_outputs)
        )
        self.linear_shortcut = nn.Linear(num_features, num_outputs)
        
        # 自适应融合模块
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * num_outputs)
        )
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.linear_shortcut.weight, 0.0)
        nn.init.constant_(self.linear_shortcut.bias, 0.0)
        nn.init.constant_(self.fusion_gate[-1].bias, 0.0)
        
    def forward(self, x, return_weights=False):
        batch_size = x.size(0)
        res_linear = self.linear_shortcut(x)
        
        numerical_tokens = self.numerical_tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_deep = torch.cat([cls_tokens, numerical_tokens], dim=1)
        x_deep = x_deep + self.pos_embedding
        x_deep = self.dropout_layer(x_deep)
        for layer in self.transformer_layers:
            x_deep = layer(x_deep)
        cls_output = x_deep[:, 0, :]
        res_deep = self.output_head(cls_output)
        
        # 计算动态权重
        gate_logits = self.fusion_gate(cls_output)                     # (batch, 2*num_outputs)
        gate_logits = gate_logits.view(batch_size, 2, self.num_outputs)
        gate_weights = torch.softmax(gate_logits, dim=1)               # (batch, 2, num_outputs)
        deep_weight = gate_weights[:, 0, :]      # (batch, num_outputs)
        linear_weight = gate_weights[:, 1, :]    # (batch, num_outputs)
        
        output = deep_weight * res_deep + linear_weight * res_linear
        
        if return_weights:
            return output, deep_weight, linear_weight
        else:
            return output

# ==================== 损失函数与训练准备 ====================
def combined_loss(y_true, y_pred):
    return nn.MSELoss()(y_true, y_pred)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FTTransformer(num_features=X_train.shape[1], num_outputs=5,
                      d_model=128, nhead=8, num_layers=10,
                      dim_feedforward=512, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=15, min_lr=1e-8, verbose=True)

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=epochs):
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    train_maes = []
    val_maes = []
    learning_rates = []
    
    # 记录每个epoch验证集上的平均融合权重（每个输出分量）
    deep_weights_history = []
    linear_weights_history = []
    
    best_val_rmse = float('inf')
    patience = 3000
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        train_mae = 0.0
        train_mse = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = combined_loss(batch_y, y_pred)
            mse = nn.MSELoss()(y_pred, batch_y)
            rmse = torch.sqrt(mse)
            mae = nn.L1Loss()(y_pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_rmse += rmse.item()
            train_mae += mae.item()
            train_mse += mse.item()

        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        all_deep_weights = []
        all_linear_weights = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, deep_w, linear_w = model(batch_X, return_weights=True)
                loss = combined_loss(batch_y, outputs)
                mse = nn.MSELoss()(outputs, batch_y)
                rmse = torch.sqrt(mse)
                mae = nn.L1Loss()(outputs, batch_y)
                val_loss += loss.item()
                val_rmse += rmse.item()
                val_mae += mae.item()
                val_mse += mse.item()
                all_deep_weights.append(deep_w.cpu().numpy())
                all_linear_weights.append(linear_w.cpu().numpy())
        
        # 计算验证集上的平均权重（每个分量）
        all_deep_weights = np.vstack(all_deep_weights)
        all_linear_weights = np.vstack(all_linear_weights)
        mean_deep_weights = all_deep_weights.mean(axis=0)
        mean_linear_weights = all_linear_weights.mean(axis=0)
        deep_weights_history.append(mean_deep_weights.copy())
        linear_weights_history.append(mean_linear_weights.copy())
        
        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)
        train_mae /= len(train_loader)
        train_mse /= len(train_loader)
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        learning_rates.append(current_lr)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('早停触发')
            break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, '
                  f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, '
                  f'Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, LR: {current_lr:.6f}')
            print(f"  平均Deep权重: {mean_deep_weights}")
            print(f"  平均Linear权重: {mean_linear_weights}")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'learning_rates': learning_rates,
        'deep_weights_history': np.array(deep_weights_history),
        'linear_weights_history': np.array(linear_weights_history),
    }
    model.load_state_dict(torch.load('best_model.pth'))
    return history

# ==================== 评估函数 ====================
def evaluate_model_with_weights(model, data_loader, scaler_y, device):
    model.eval()
    all_preds = []
    all_trues = []
    all_deep_weights = []
    all_linear_weights = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs, deep_w, linear_w = model(batch_X, return_weights=True)
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_deep_weights.append(deep_w.cpu().numpy())
            all_linear_weights.append(linear_w.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    all_deep_weights = np.vstack(all_deep_weights)
    all_linear_weights = np.vstack(all_linear_weights)

    all_preds_orig = scaler_y.inverse_transform(all_preds)
    all_trues_orig = scaler_y.inverse_transform(all_trues)

    total_mse = mean_squared_error(all_trues_orig, all_preds_orig)
    total_rmse = np.sqrt(total_mse)
    total_mae = mean_absolute_error(all_trues_orig, all_preds_orig)
    total_r2 = r2_score(all_trues_orig, all_preds_orig)
    mse_loss = np.mean((all_trues_orig - all_preds_orig) ** 2)
    mae_loss = np.mean(np.abs(all_trues_orig - all_preds_orig))
    total_combined = mse_loss + 0.33 * mae_loss

    n_outputs = all_trues_orig.shape[1]
    component_metrics = {}
    for i in range(n_outputs):
        y_true_i = all_trues_orig[:, i]
        y_pred_i = all_preds_orig[:, i]
        rmse_i = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
        mae_i = mean_absolute_error(y_true_i, y_pred_i)
        r2_i = r2_score(y_true_i, y_pred_i)
        epsilon = 1e-8
        mre_i = np.mean(np.abs((y_true_i - y_pred_i) / (np.abs(y_true_i) + epsilon)))
        component_metrics[i] = {'RMSE': rmse_i, 'MAE': mae_i, 'R2': r2_i, 'MRE': mre_i}

    overall_metrics = {
        'MSE': total_mse,
        'RMSE': total_rmse,
        'MAE': total_mae,
        'R2': total_r2,
        'Combined_Loss': total_combined
    }
    
    mean_deep_weights = all_deep_weights.mean(axis=0)
    mean_linear_weights = all_linear_weights.mean(axis=0)
    
    return overall_metrics, component_metrics, (all_trues_orig, all_preds_orig), (mean_deep_weights, mean_linear_weights)

# ==================== 绘图函数 ====================
def plot_alpha_curves_separate(alpha, true, pred, target_names, save_prefix='alpha_curve'):
    """攻角-预测曲线，每个输出分量独立画布"""
    sort_idx = np.argsort(alpha)
    alpha_sorted = alpha[sort_idx]
    true_sorted = true[sort_idx, :]
    pred_sorted = pred[sort_idx, :]
    for i, name in enumerate(target_names):
        plt.figure(figsize=(8, 6))
        plt.plot(alpha_sorted, true_sorted[:, i], 'b-', label='真实值', linewidth=2)
        plt.plot(alpha_sorted, pred_sorted[:, i], 'r--', label='预测值', linewidth=2)
        plt.xlabel('攻角 (°)', fontsize=18)
        plt.ylabel(name, fontsize=18)
        plt.title(f'{name} 随攻角变化', fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        save_path = f"{save_prefix}_{name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"攻角曲线图已保存至 {save_path}")

def plot_true_vs_pred(true, pred, target_names, component_metrics, save_path='component_comparison.png'):
    """预测值-真实值散点图（2x3子图）"""
    n = len(target_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], alpha=0.6, edgecolors='k', s=30)
        min_val = min(true[:, i].min(), pred[:, i].min())
        max_val = max(true[:, i].max(), pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
        ax.set_xlabel('真实值', fontsize=14)
        ax.set_ylabel('预测值', fontsize=14)
        ax.set_title(f'{name}\n(R² = {component_metrics[i]["R2"]:.3f})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    if n < len(axes):
        axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"散点对比图已保存至 {save_path}")

def plot_weight_evolution_separate(deep_weights_history, linear_weights_history, target_names, save_prefix='weight_evolution'):
    """
    融合权重变化图：每个输出分量单独绘制一个独立画布，大小与攻角曲线一致（8×6英寸），字号统一。
    deep_weights_history: (n_epochs, num_outputs)
    linear_weights_history: (n_epochs, num_outputs)
    """
    epochs = deep_weights_history.shape[0]
    num_outputs = deep_weights_history.shape[1]
    for i in range(num_outputs):
        plt.figure(figsize=(8, 6))
        plt.plot(range(epochs), deep_weights_history[:, i], 'b-', label='Deep分支权重', linewidth=2)
        plt.plot(range(epochs), linear_weights_history[:, i], 'r-', label='Linear分支权重', linewidth=2)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('平均权重', fontsize=18)
        plt.title(f'{target_names[i]} 融合权重变化', fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        save_path = f"{save_prefix}_{target_names[i]}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"权重演化图已保存至 {save_path}")

# ==================== 训练模型 ====================
print('开始训练...')
history = train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=epochs)
print('训练完成，加载最佳模型')

# ==================== 可视化权重演化（独立画布） ====================
deep_hist = history['deep_weights_history']   # (n_epochs, 5)
linear_hist = history['linear_weights_history']
plot_weight_evolution_separate(deep_hist, linear_hist, target_columns, save_prefix='weight_evolution')

# ==================== 评估并输出分量指标及最终权重数值 ====================
print("\n===== 测试集评估结果（原始尺度） =====")
test_overall, test_component, (test_true, test_pred), (test_deep_w, test_linear_w) = evaluate_model_with_weights(model, test_loader, scaler_y, device)

print(f"总体指标: MSE={test_overall['MSE']:.4f}, RMSE={test_overall['RMSE']:.4f}, MAE={test_overall['MAE']:.4f}, R2={test_overall['R2']:.4f}, Combined={test_overall['Combined_Loss']:.4f}")

print("\n各分量详细指标:")
for i, name in enumerate(target_columns):
    m = test_component[i]
    print(f"{name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, MRE={m['MRE']:.4%}")

print("\n===== 测试集最终融合权重（平均） =====")
print("Deep分支权重（每个输出分量）:")
for i, name in enumerate(target_columns):
    print(f"  {name}: {test_deep_w[i]:.4f}")
print("Linear分支权重（每个输出分量）:")
for i, name in enumerate(target_columns):
    print(f"  {name}: {test_linear_w[i]:.4f}")

print("\n===== 训练集评估结果 =====")
train_overall, train_component, _, _ = evaluate_model_with_weights(model, train_loader, scaler_y, device)
print(f"总体RMSE={train_overall['RMSE']:.4f}, R2={train_overall['R2']:.4f}")

print("\n===== 验证集评估结果 =====")
val_overall, val_component, _, _ = evaluate_model_with_weights(model, val_loader, scaler_y, device)
print(f"总体RMSE={val_overall['RMSE']:.4f}, R2={val_overall['R2']:.4f}")

# ==================== 原有可视化 ====================
plot_true_vs_pred(test_true, test_pred, target_columns, test_component, save_path='component_comparison.png')
test_alpha = X_test[:, feature_columns.index('AA')]
plot_alpha_curves_separate(test_alpha, test_true, test_pred, target_columns, save_prefix='alpha_curve')

print("\n所有任务完成！")


