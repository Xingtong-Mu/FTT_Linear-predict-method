# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ==================== 中文字体设置 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 随机种子 ====================
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ==================== 参数定义 ====================
epochs = 300
batch_size = 64
learning_rate = 0.001
weight_decay = 0.0001

# ==================== 数据加载与预处理 ====================
print("开始加载数据...")

df = pd.read_excel('data_all.xlsx')

feature_columns = ['MA', 'BETA', 'AA']
target_columns = ['CL', 'CD', 'CN', 'CA', 'CM']

X = df[feature_columns].values
y = df[target_columns].values

print(f"特征维度: {X.shape}, 目标维度: {y.shape}")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# ==================== 转换为 Tensor ====================
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val_scaled)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ==================== SAINT 模型模块 ====================
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SAINTBlock(nn.Module):
    """
    SAINT 基本模块：
    1. Column Attention：在特征 token 维度上做注意力；
    2. Row Attention：在 batch 样本维度上做注意力；
    3. 前馈网络。
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        # 列注意力：学习同一个样本内部不同特征之间的关系
        self.col_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.col_norm1 = nn.LayerNorm(d_model)
        self.col_ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.col_norm2 = nn.LayerNorm(d_model)

        # 行注意力：学习同一个 batch 内不同样本之间的关系
        self.row_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.row_norm1 = nn.LayerNorm(d_model)
        self.row_ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.row_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch_size, num_tokens, d_model)

        # ==================== Column Attention ====================
        col_out, _ = self.col_attn(x, x, x)
        x = self.col_norm1(x + col_out)

        col_ffn_out = self.col_ffn(x)
        x = self.col_norm2(x + col_ffn_out)

        # ==================== Row Attention ====================
        # 原始 x: (batch, tokens, d_model)
        # 转换为: (tokens, batch, d_model)
        # 对每个 token 位置，在 batch 样本之间做注意力
        x_row = x.transpose(0, 1)

        row_out, _ = self.row_attn(x_row, x_row, x_row)
        x_row = self.row_norm1(x_row + row_out)

        row_ffn_out = self.row_ffn(x_row)
        x_row = self.row_norm2(x_row + row_ffn_out)

        # 转回: (batch, tokens, d_model)
        x = x_row.transpose(0, 1)

        return x


class NumericalSAINT(nn.Module):
    """
    数值型 SAINT 模型。

    说明：
    原始 SAINT 同时支持类别特征和连续特征。
    由于当前气动力数据只有连续变量 MA、BETA、AA，
    因此这里将每个连续变量映射为特征 token，
    再通过 Column Attention 和 Row Attention 建模。
    """

    def __init__(
        self,
        num_features,
        num_outputs,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        # 连续特征 token 化
        # 每个特征拥有独立的权重和偏置
        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model)
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        # 列嵌入，用于区分 MA、BETA、AA
        self.column_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model)
        )

        # CLS token，用于聚合全局信息
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.dropout = nn.Dropout(dropout)

        # 多层 SAINT Block
        self.blocks = nn.ModuleList([
            SAINTBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_outputs)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.normal_(self.column_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch_size, num_features)

        batch_size = x.size(0)

        # 连续变量 token 化
        # x -> (batch, num_features, 1)
        x = x.unsqueeze(-1)

        # 每个特征使用独立参数映射到 d_model 维
        x = x * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)

        # 加入列嵌入
        x = x + self.column_embedding
        x = self.dropout(x)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # SAINT 编码
        for block in self.blocks:
            x = block(x)

        # 取 CLS token 作为全局表征
        cls_output = x[:, 0, :]

        # 多输出回归
        out = self.output_head(cls_output)

        return out

# ==================== 损失函数 ====================
def combined_loss(y_true, y_pred):
    mse = nn.MSELoss()(y_pred, y_true)
    return mse

# ==================== 模型、优化器、学习率调度器 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")

model = NumericalSAINT(
    num_features=X_train.shape[1],
    num_outputs=len(target_columns),
    d_model=64,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=15,
    min_lr=1e-8
)

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=epochs):
    train_losses = []
    val_losses = []

    train_rmses = []
    val_rmses = []

    train_maes = []
    val_maes = []

    learning_rates = []

    best_val_rmse = float('inf')
    patience = 3000
    patience_counter = 0

    best_model_path = 'best_saint.pth'

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_rmse = 0.0
        train_mae = 0.0
        train_mse = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

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

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)

                loss = combined_loss(batch_y, outputs)
                mse = nn.MSELoss()(outputs, batch_y)
                rmse = torch.sqrt(mse)
                mae = nn.L1Loss()(outputs, batch_y)

                val_loss += loss.item()
                val_rmse += rmse.item()
                val_mae += mae.item()
                val_mse += mse.item()

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
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('早停触发')
            break

        if (epoch + 1) % 20 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Train RMSE: {train_rmse:.4f}, '
                f'Val RMSE: {val_rmse:.4f}, '
                f'Train MAE: {train_mae:.4f}, '
                f'Val MAE: {val_mae:.4f}, '
                f'Train MSE: {train_mse:.4f}, '
                f'Val MSE: {val_mse:.4f}, '
                f'LR: {current_lr:.8f}'
            )

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'train_maes': train_maes,
        'val_maes': val_maes,
        'learning_rates': learning_rates
    }

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return history

# ==================== 评估函数 ====================
def evaluate_model(model, data_loader, scaler_y, device):
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

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

    return overall_metrics, component_metrics, (all_trues_orig, all_preds_orig)

# ==================== 开始训练 ====================
print('开始训练 SAINT...')

history = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs=epochs
)

print('训练完成，加载最佳模型')

# ==================== 测试集评估 ====================
print("\n===== 测试集评估结果（原始尺度） =====")

test_overall, test_component, (test_true, test_pred) = evaluate_model(
    model,
    test_loader,
    scaler_y,
    device
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

# ==================== 训练集评估 ====================
print("\n===== 训练集评估结果 =====")

train_overall, _, _ = evaluate_model(
    model,
    train_loader,
    scaler_y,
    device
)

print(
    f"总体RMSE={train_overall['RMSE']:.4f}, "
    f"R2={train_overall['R2']:.4f}"
)

# ==================== 验证集评估 ====================
print("\n===== 验证集评估结果 =====")

val_overall, _, _ = evaluate_model(
    model,
    val_loader,
    scaler_y,
    device
)

print(
    f"总体RMSE={val_overall['RMSE']:.4f}, "
    f"R2={val_overall['R2']:.4f}"
)

# ==================== 可视化 ====================
def plot_true_vs_pred(
    true,
    pred,
    target_names,
    component_metrics,
    save_path='saint_comparison.png'
):
    n = len(target_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, name in enumerate(target_names):
        ax = axes[i]

        ax.scatter(
            true[:, i],
            pred[:, i],
            alpha=0.6,
            edgecolors='k',
            s=30
        )

        min_val = min(true[:, i].min(), pred[:, i].min())
        max_val = max(true[:, i].max(), pred[:, i].max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            lw=2,
            label='理想线'
        )

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

    print(f"对比图已保存至 {save_path}")

plot_true_vs_pred(
    test_true,
    test_pred,
    target_columns,
    test_component,
    save_path='saint_comparison.png'
)

print("\nSAINT 对比实验完成！")



