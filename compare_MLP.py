# %%
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import copy
import random
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")
# ==================== 参数设置 ====================
# DATA_FILE = Path("data_low.csv")
# FEATURE_COLUMNS = ["MA", "BETA", "AA"]
# TARGET_COLUMNS = ["CL", "CM", "CD"]
N_SPLITS = 5
INNER_VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 100
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
# ==================== 随机种子 ====================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(RANDOM_SEED)
# ==================== 字段名兼容 ====================
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "MA": ["MA", "Mach", "MACH", "mach", "M0", "Mo", "MO"],
        "Re": ["Re", "RE", "re", "Reynolds", "ReynoldsNumber", "Reynolds_Number"],
        "BETA": ["BETA", "Beta", "beta", "Sideslip", "Sideslip_deg"],
        "AA": ["AA", "Alpha_deg", "ALPHA", "Alpha", "alpha", "AOA", "AoA"],
        "CL": ["CL", "Cl", "cl", "CZI", "Czi", "czi"],
        "CM": ["CM", "Cm", "cm", "CMI25", "Cmi25", "cmi25"],
        "CD": ["CD", "Cd", "cd"]
    }
    df = df.copy()
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    lookup = {str(column).casefold(): column for column in df.columns}
    rename_mapping = {}
    for standard_name, possible_names in aliases.items():
        if standard_name in df.columns:
            continue
        for candidate in possible_names:
            original_name = lookup.get(candidate.casefold())
            if original_name is not None:
                rename_mapping[original_name] = standard_name
                break
    return df.rename(columns=rename_mapping)
# ==================== CSV读取 ====================
def read_csv_robust(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "cp1252", "latin1"]
    separators = [",", "\t", ";", None]
    last_error = None
    for encoding in encodings:
        for separator in separators:
            try:
                if separator is None:
                    df = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
                else:
                    df = pd.read_csv(file_path, sep=separator, encoding=encoding)
                if len(df.columns) > 1:
                    return df
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"无法读取文件：{file_path}\n最后一次错误：{last_error}")
# ==================== 数据加载 ====================
def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{file_path.resolve()}")
    print("=" * 80)
    print("Loading RAE2822 dataset")
    print("=" * 80)
    print(f"File: {file_path.resolve()}")
    df = read_csv_robust(file_path)
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    df = normalize_column_names(df)
    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要字段：{missing_columns}\n实际字段：{list(df.columns)}")
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    original_rows = len(df)
    df = df.dropna(subset=required_columns).copy()
    invalid_rows = original_rows - len(df)
    duplicate_mask = df.duplicated(subset=required_columns, keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy().reset_index(drop=True)
    df["DATA_INDEX"] = np.arange(len(df))
    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Output targets: {TARGET_COLUMNS}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    return df
# ==================== MLP模型 ====================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.shortcut = nn.Linear(input_dim, output_dim)
        self.initialize_weights()
    def initialize_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.shortcut.weight)
        nn.init.zeros_(self.shortcut.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + self.shortcut(x)
# ==================== DataLoader ====================
def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    if len(dataset) == 0:
        raise ValueError("无法创建DataLoader：数据集为空。")
    actual_batch_size = min(batch_size, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_last = bool(shuffle and len(dataset) > actual_batch_size and len(dataset) % actual_batch_size == 1)
    return DataLoader(dataset, batch_size=actual_batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None, drop_last=drop_last)
# ==================== 单折训练 ====================
def train_one_fold(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, fold_number: int) -> tuple[nn.Module, pd.DataFrame]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-8)
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    history_records = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            current_batch_size = batch_X.size(0)
            train_loss_sum += loss.item() * current_batch_size
            train_sample_count += current_batch_size
        if train_sample_count == 0:
            raise RuntimeError(f"Fold {fold_number}训练集未产生有效批次。")
        train_loss = train_loss_sum / train_sample_count
        model.eval()
        validation_loss_sum = 0.0
        validation_sample_count = 0
        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                current_batch_size = batch_X.size(0)
                validation_loss_sum += loss.item() * current_batch_size
                validation_sample_count += current_batch_size
        if validation_sample_count == 0:
            raise RuntimeError(f"Fold {fold_number}独立验证集未产生有效批次。")
        validation_loss = validation_loss_sum / validation_sample_count
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history_records.append({"Fold": fold_number, "Epoch": epoch, "Train_Loss": train_loss, "Validation_Loss": validation_loss, "Learning_Rate": current_lr})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS}, Train Loss={train_loss:.6f}, Independent Validation Loss={validation_loss:.6f}, LR={current_lr:.8f}, Patience={patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}")
            break
    model.load_state_dict(best_state)
    print(f"Best epoch={best_epoch}, best independent validation loss={best_validation_loss:.6f}")
    return model, pd.DataFrame(history_records)
# ==================== 模型预测 ====================
def predict_model(model: nn.Module, data_loader: DataLoader, scaler_y: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions_scaled = []
    true_values_scaled = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            predictions = model(batch_X)
            predictions_scaled.append(predictions.cpu().numpy())
            true_values_scaled.append(batch_y.numpy())
    if not predictions_scaled:
        raise RuntimeError("预测DataLoader为空，无法生成预测结果。")
    predictions_scaled = np.vstack(predictions_scaled)
    true_values_scaled = np.vstack(true_values_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    true_values = scaler_y.inverse_transform(true_values_scaled)
    return true_values, predictions
# ==================== 分量指标 ====================
def calculate_component_metrics(y_true: np.ndarray, y_pred: np.ndarray, fold_number: int) -> list[dict]:
    records = []
    epsilon = 1e-8
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        true_component = y_true[:, target_index]
        predicted_component = y_pred[:, target_index]
        mse = mean_squared_error(true_component, predicted_component)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_component, predicted_component)
        r2 = r2_score(true_component, predicted_component)
        mre = np.mean(np.abs((true_component - predicted_component) / (np.abs(true_component) + epsilon))) * 100.0
        smape = np.mean(2.0 * np.abs(predicted_component - true_component) / (np.abs(true_component) + np.abs(predicted_component) + epsilon)) * 100.0
        records.append({"Fold": fold_number, "Target": target_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MRE_percent": mre, "SMAPE_percent": smape})
    return records
# ==================== 五折外层交叉验证 ====================
def run_five_fold_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_all = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    oof_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_fold = np.full(len(df), -1, dtype=int)
    metrics_records = []
    history_frames = []
    split_records = []
    outer_splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    print()
    print("=" * 80)
    print("MLP nested cross-validation method")
    print("Outer split: shuffled five-fold KFold")
    print("Inner split: 20% of each outer-training fold used as independent validation")
    print(f"Approximate global training ratio: {(1.0 - 1.0 / N_SPLITS) * (1.0 - INNER_VALIDATION_RATIO) * 100.0:.1f}%")
    print(f"Approximate global validation ratio: {(1.0 - 1.0 / N_SPLITS) * INNER_VALIDATION_RATIO * 100.0:.1f}%")
    print(f"Approximate global outer-test ratio: {100.0 / N_SPLITS:.1f}%")
    print(f"Number of samples: {len(df)}")
    print(f"Number of outer folds: {N_SPLITS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Hidden dimensions: {HIDDEN_DIMS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 80)
    for fold_number, (outer_train_index, outer_test_index) in enumerate(outer_splitter.split(X_all), start=1):
        print()
        print("=" * 80)
        print(f"MLP Outer Fold {fold_number}/{N_SPLITS}")
        print("=" * 80)
        fold_seed = RANDOM_SEED + fold_number
        set_random_seed(fold_seed)
        inner_train_index, inner_validation_index = train_test_split(outer_train_index, test_size=INNER_VALIDATION_RATIO, shuffle=True, random_state=fold_seed)
        overlap_train_validation = np.intersect1d(inner_train_index, inner_validation_index)
        overlap_train_test = np.intersect1d(inner_train_index, outer_test_index)
        overlap_validation_test = np.intersect1d(inner_validation_index, outer_test_index)
        if len(overlap_train_validation) > 0 or len(overlap_train_test) > 0 or len(overlap_validation_test) > 0:
            raise RuntimeError(f"Fold {fold_number}数据划分存在索引重复。")
        if len(np.union1d(np.union1d(inner_train_index, inner_validation_index), outer_test_index)) != len(df):
            raise RuntimeError(f"Fold {fold_number}数据划分未覆盖全部样本。")
        X_train = X_all[inner_train_index]
        X_validation = X_all[inner_validation_index]
        X_test = X_all[outer_test_index]
        y_train = y_all[inner_train_index]
        y_validation = y_all[inner_validation_index]
        y_test = y_all[outer_test_index]
        train_ratio = len(inner_train_index) / len(df) * 100.0
        validation_ratio = len(inner_validation_index) / len(df) * 100.0
        test_ratio = len(outer_test_index) / len(df) * 100.0
        print(f"Inner training samples: {len(inner_train_index)} ({train_ratio:.2f}% of global data)")
        print(f"Independent validation samples: {len(inner_validation_index)} ({validation_ratio:.2f}% of global data)")
        print(f"Outer test samples: {len(outer_test_index)} ({test_ratio:.2f}% of global data)")
        print("Index overlap check: passed")
        split_records.append({"Fold": fold_number, "Inner_Train_Samples": len(inner_train_index), "Independent_Validation_Samples": len(inner_validation_index), "Outer_Test_Samples": len(outer_test_index), "Inner_Train_Global_Percent": train_ratio, "Independent_Validation_Global_Percent": validation_ratio, "Outer_Test_Global_Percent": test_ratio, "Train_Validation_Overlap": len(overlap_train_validation), "Train_Test_Overlap": len(overlap_train_test), "Validation_Test_Overlap": len(overlap_validation_test)})
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_validation_scaled = scaler_X.transform(X_validation)
        y_validation_scaled = scaler_y.transform(y_validation)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
        train_loader = create_data_loader(X_train_scaled, y_train_scaled, BATCH_SIZE, True, fold_seed)
        validation_loader = create_data_loader(X_validation_scaled, y_validation_scaled, BATCH_SIZE, False, fold_seed)
        test_loader = create_data_loader(X_test_scaled, y_test_scaled, BATCH_SIZE, False, fold_seed)
        model = SimpleMLP(input_dim=len(FEATURE_COLUMNS), output_dim=len(TARGET_COLUMNS), hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(DEVICE)
        model, fold_history = train_one_fold(model, train_loader, validation_loader, fold_number)
        history_frames.append(fold_history)
        y_true, y_pred = predict_model(model, test_loader, scaler_y)
        if not np.allclose(y_true, y_test, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"Fold {fold_number}外层测试集逆标准化结果与原始数据不一致。")
        oof_predictions[outer_test_index] = y_pred
        oof_fold[outer_test_index] = fold_number
        fold_metrics = calculate_component_metrics(y_true, y_pred, fold_number)
        metrics_records.extend(fold_metrics)
        print("Outer-test metrics:")
        for metric in fold_metrics:
            print(f"{metric['Target']}: RMSE={metric['RMSE']:.6f}, MAE={metric['MAE']:.6f}, R2={metric['R2']:.6f}, MRE={metric['MRE_percent']:.3f}%, SMAPE={metric['SMAPE_percent']:.3f}%")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "fold": fold_number,
            "random_seed": fold_seed,
            "inner_validation_ratio": INNER_VALIDATION_RATIO,
            "feature_scaler_mean": scaler_X.mean_,
            "feature_scaler_scale": scaler_X.scale_,
            "target_scaler_mean": scaler_y.mean_,
            "target_scaler_scale": scaler_y.scale_,
            "inner_train_sample_count": len(inner_train_index),
            "independent_validation_sample_count": len(inner_validation_index),
            "outer_test_sample_count": len(outer_test_index)
        }
        torch.save(checkpoint, f"best_mlp_fold_{fold_number}.pth")
        del model
        del train_loader
        del validation_loader
        del test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(oof_predictions).any():
        raise RuntimeError("OOF预测中存在缺失值，请检查五折交叉验证执行过程。")
    if np.any(oof_fold < 1):
        raise RuntimeError("部分样本未被分配到外层测试折。")
    metrics_df = pd.DataFrame(metrics_records)
    history_df = pd.concat(history_frames, ignore_index=True)
    split_df = pd.DataFrame(split_records)
    summary_df = metrics_df.groupby("Target")[["MSE", "RMSE", "MAE", "R2", "MRE_percent", "SMAPE_percent"]].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in summary_df.columns]
    summary_df = summary_df.reset_index()
    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        oof_df[f"{target_name}_True"] = y_all[:, target_index]
        oof_df[f"{target_name}_Pred"] = oof_predictions[:, target_index]
        oof_df[f"{target_name}_Error"] = oof_predictions[:, target_index] - y_all[:, target_index]
        oof_df[f"{target_name}_Absolute_Error"] = np.abs(oof_predictions[:, target_index] - y_all[:, target_index])
    return metrics_df, summary_df, oof_df, history_df, split_df
# ==================== OOF总体指标 ====================
def calculate_oof_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    epsilon = 1e-8
    for target_name in TARGET_COLUMNS:
        y_true = oof_df[f"{target_name}_True"].to_numpy()
        y_pred = oof_df[f"{target_name}_Pred"].to_numpy()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mre = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100.0
        records.append({"Target": target_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MRE_percent": mre, "SMAPE_percent": smape})
    return pd.DataFrame(records)
# ==================== 真实值-预测值散点图 ====================
def plot_oof_true_vs_pred(oof_df: pd.DataFrame) -> None:
    for target_name in TARGET_COLUMNS:
        true_values = oof_df[f"{target_name}_True"].to_numpy()
        predicted_values = oof_df[f"{target_name}_Pred"].to_numpy()
        minimum_value = min(true_values.min(), predicted_values.min())
        maximum_value = max(true_values.max(), predicted_values.max())
        r2 = r2_score(true_values, predicted_values)
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, alpha=0.7, s=35, edgecolors="black", linewidths=0.4)
        plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], linestyle="--", linewidth=2, label="Ideal prediction")
        plt.xlabel(f"True {target_name}", fontsize=16)
        plt.ylabel(f"Predicted {target_name}", fontsize=16)
        plt.title(f"MLP Five-fold OOF Prediction for {target_name}\n$R^2$ = {r2:.4f}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"mlp_cv_oof_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 每折指标图 ====================
def plot_fold_metrics(metrics_df: pd.DataFrame) -> None:
    for target_name in TARGET_COLUMNS:
        target_metrics = metrics_df.loc[metrics_df["Target"] == target_name].sort_values("Fold")
        plt.figure(figsize=(8, 6))
        plt.plot(target_metrics["Fold"], target_metrics["RMSE"], marker="o", linewidth=2, label="RMSE")
        plt.plot(target_metrics["Fold"], target_metrics["MAE"], marker="s", linewidth=2, label="MAE")
        plt.xlabel("Outer fold", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title(f"MLP Outer-test Error Metrics for {target_name}", fontsize=17)
        plt.xticks(range(1, N_SPLITS + 1))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"mlp_cv_fold_metrics_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 各折训练历史图 ====================
def plot_training_history(history_df: pd.DataFrame) -> None:
    for fold_number in range(1, N_SPLITS + 1):
        fold_history = history_df.loc[history_df["Fold"] == fold_number]
        if fold_history.empty:
            continue
        plt.figure(figsize=(8, 6))
        plt.plot(fold_history["Epoch"], fold_history["Train_Loss"], linewidth=2, label="Training loss")
        plt.plot(fold_history["Epoch"], fold_history["Validation_Loss"], linewidth=2, label="Independent validation loss")
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Standardized MSE loss", fontsize=16)
        plt.title(f"MLP Training History - Outer Fold {fold_number}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"mlp_training_history_fold_{fold_number}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 输出五折汇总 ====================
def print_cv_summary(summary_df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("MLP five-fold outer-test cross-validation summary")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        target_name = row["Target"]
        print()
        print(f"Target: {target_name}")
        print(f"MSE: {row['MSE_mean']:.6f} ± {row['MSE_std']:.6f}")
        print(f"RMSE: {row['RMSE_mean']:.6f} ± {row['RMSE_std']:.6f}")
        print(f"MAE: {row['MAE_mean']:.6f} ± {row['MAE_std']:.6f}")
        print(f"R2: {row['R2_mean']:.6f} ± {row['R2_std']:.6f}")
        print(f"MRE: {row['MRE_percent_mean']:.3f}% ± {row['MRE_percent_std']:.3f}%")
        print(f"SMAPE: {row['SMAPE_percent_mean']:.3f}% ± {row['SMAPE_percent_std']:.3f}%")
# ==================== 主程序 ====================
def main() -> None:
    df = load_dataset(DATA_FILE)
    metrics_df, summary_df, oof_df, history_df, split_df = run_five_fold_cv(df)
    oof_metrics_df = calculate_oof_metrics(oof_df)
    metrics_df.to_csv("mlp_cv_metrics_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_df.to_csv("mlp_cv_metrics_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_df.to_csv("mlp_cv_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    history_df.to_csv("mlp_cv_training_history.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_metrics_df.to_csv("mlp_cv_oof_overall_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    split_df.to_csv("mlp_cv_data_split_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    print_cv_summary(summary_df)
    print()
    print("=" * 80)
    print("MLP overall out-of-fold metrics")
    print("=" * 80)
    print(oof_metrics_df.to_string(index=False))
    print()
    print("=" * 80)
    print("MLP data split summary")
    print("=" * 80)
    print(split_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)
    print("mlp_cv_metrics_each_fold.csv")
    print("mlp_cv_metrics_summary.csv")
    print("mlp_cv_oof_predictions.csv")
    print("mlp_cv_oof_overall_metrics.csv")
    print("mlp_cv_training_history.csv")
    print("mlp_cv_data_split_summary.csv")
    print("best_mlp_fold_1.pth ... best_mlp_fold_5.pth")
    for target_name in TARGET_COLUMNS:
        print(f"mlp_cv_oof_{target_name}.png")
    for target_name in TARGET_COLUMNS:
        print(f"mlp_cv_fold_metrics_{target_name}.png")
    print("mlp_training_history_fold_1.png ... mlp_training_history_fold_5.png")
    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_training_history(history_df)
    print()
    print("All MLP nested five-fold cross-validation tasks completed.")
if __name__ == "__main__":
    main()
