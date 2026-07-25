# %%
import time
# -*- coding: utf-8 -*-
from __future__ import annotations
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")
# ==================== 参数设置 ====================
#路径请自行指定，例如：DATA_FILE = Path("data_2822.csv")
# DATA_FILE = Path("data_2822.csv")
# FEATURE_COLUMNS = ["MA", "BETA", "AA"]
# TARGET_COLUMNS = ["CL", "CM", "CD"]
N_SPLITS = 5
RANDOM_SEED = 42
INNER_VALIDATION_RATIO = 0.20
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 100
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LR_SCHEDULER_PATIENCE = 15
RIDGE_ALPHA = 1.0
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
# ==================== 字段名称兼容 ====================
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {"MA": ["Mach", "M0", "Mo", "MO", "MA", "mach"], "AA": ["Alpha_deg", "ALPHA", "Alpha", "alpha", "AA", "AOA"], "Re": ["Re", "RE", "re", "Reynolds", "ReynoldsNumber"], "CL": ["CL", "CZI", "Czi", "czi"], "CM": ["Cm", "CM", "CMI25", "Cmi25", "cmi25"], "BETA": ["Beta_deg", "BETA", "Beta", "beta", "BETA"], "CD": ["CD", "CD25", "cd25"]}
    lookup = {str(column).replace("\ufeff", "").strip().casefold(): column for column in df.columns}
    rename_mapping = {}
    for standard_name, possible_names in aliases.items():
        if standard_name in df.columns:
            continue
        for candidate in possible_names:
            candidate_key = candidate.casefold()
            if candidate_key in lookup:
                rename_mapping[lookup[candidate_key]] = standard_name
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
    raise RuntimeError(f"无法读取数据文件：{file_path}\n最后一次错误：{last_error}")
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
    if len(df) < N_SPLITS:
        raise ValueError(f"有效样本数为{len(df)}，无法进行{N_SPLITS}折交叉验证。")
    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint output targets: {TARGET_COLUMNS}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    return df
# ==================== ReGLU激活函数 ====================
class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_half, second_half = x.chunk(2, dim=-1)
        return first_half * torch.relu(second_half)
# ==================== 数值特征Tokenizer ====================
class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model))
        self.bias = nn.Parameter(torch.randn(num_features, d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
# ==================== ReGLU Transformer层 ====================
class TransformerEncoderLayerReGLU(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = ReGLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attn(src, src, src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(attention_output))
        feedforward_output = self.linear1(src)
        feedforward_output = self.activation(feedforward_output)
        feedforward_output = self.dropout(feedforward_output)
        feedforward_output = self.linear2(feedforward_output)
        src = self.norm2(src + self.dropout2(feedforward_output))
        return src
# ==================== 小型FT-Transformer与自适应门控 ====================
class ReducedFTTransformer(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.numerical_tokenizer = NumericalFeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))
        self.dropout_layer = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayerReGLU(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_outputs))
        self.linear_shortcut = nn.Linear(num_features, num_outputs)
        self.fusion_gate = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2 * num_outputs))
        self.initialize_weights()
    def initialize_weights(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.linear_shortcut.weight)
        nn.init.zeros_(self.linear_shortcut.bias)
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.zeros_(self.fusion_gate[-1].bias)
    def forward(self, x: torch.Tensor, return_details: bool = False):
        batch_size = x.size(0)
        linear_result = self.linear_shortcut(x)
        numerical_tokens = self.numerical_tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        deep_input = torch.cat([cls_tokens, numerical_tokens], dim=1)
        deep_input = self.dropout_layer(deep_input + self.pos_embedding)
        for layer in self.transformer_layers:
            deep_input = layer(deep_input)
        cls_output = deep_input[:, 0, :]
        deep_result = self.output_head(cls_output)
        gate_logits = self.fusion_gate(cls_output)
        gate_logits = gate_logits.view(batch_size, 2, self.num_outputs)
        gate_weights = torch.softmax(gate_logits, dim=1)
        deep_weight = gate_weights[:, 0, :]
        linear_weight = gate_weights[:, 1, :]
        output = deep_weight * deep_result + linear_weight * linear_result
        if return_details:
            return output, deep_result, linear_result, deep_weight, linear_weight
        return output
# ==================== Ridge初始化线性分支 ====================
def initialize_linear_branch_from_ridge(model: ReducedFTTransformer, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, alpha: float) -> Ridge:
    ridge_model = Ridge(alpha=alpha, fit_intercept=True)
    ridge_model.fit(X_train_scaled, y_train_scaled)
    ridge_coefficients = np.asarray(ridge_model.coef_, dtype=np.float32)
    ridge_intercepts = np.asarray(ridge_model.intercept_, dtype=np.float32)
    expected_weight_shape = tuple(model.linear_shortcut.weight.shape)
    expected_bias_shape = tuple(model.linear_shortcut.bias.shape)
    if ridge_coefficients.shape != expected_weight_shape:
        raise RuntimeError(f"Ridge系数形状错误，预期{expected_weight_shape}，实际{ridge_coefficients.shape}")
    if ridge_intercepts.shape != expected_bias_shape:
        raise RuntimeError(f"Ridge截距形状错误，预期{expected_bias_shape}，实际{ridge_intercepts.shape}")
    with torch.no_grad():
        model.linear_shortcut.weight.copy_(torch.tensor(ridge_coefficients, dtype=torch.float32, device=DEVICE))
        model.linear_shortcut.bias.copy_(torch.tensor(ridge_intercepts, dtype=torch.float32, device=DEVICE))
    return ridge_model
# ==================== 模型参数量 ====================
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
# ==================== DataLoader ====================
def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader_batch_size = min(batch_size, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(seed)
    if shuffle:
        return DataLoader(dataset, batch_size=loader_batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator, drop_last=False)
    return DataLoader(dataset, batch_size=loader_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
# ==================== 单折训练 ====================
def train_one_fold(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, fold_number: int) -> tuple[nn.Module, pd.DataFrame, int, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=LR_SCHEDULER_PATIENCE, min_lr=1e-8)
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
        train_loss = train_loss_sum / train_sample_count
        model.eval()
        validation_loss_sum = 0.0
        validation_sample_count = 0
        deep_weight_batches = []
        linear_weight_batches = []
        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                predictions, _, _, deep_weights, linear_weights = model(batch_X, return_details=True)
                loss = criterion(predictions, batch_y)
                current_batch_size = batch_X.size(0)
                validation_loss_sum += loss.item() * current_batch_size
                validation_sample_count += current_batch_size
                deep_weight_batches.append(deep_weights.cpu().numpy())
                linear_weight_batches.append(linear_weights.cpu().numpy())
        validation_loss = validation_loss_sum / validation_sample_count
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        mean_deep_weights = np.vstack(deep_weight_batches).mean(axis=0)
        mean_linear_weights = np.vstack(linear_weight_batches).mean(axis=0)
        history_record = {"Fold": fold_number, "Epoch": epoch, "Train_Loss": train_loss, "Validation_Loss": validation_loss, "Learning_Rate": current_lr}
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            history_record[f"{target_name}_Deep_Weight"] = mean_deep_weights[target_index]
            history_record[f"{target_name}_Linear_Weight"] = mean_linear_weights[target_index]
        history_records.append(history_record)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS}, Train Loss={train_loss:.6f}, Validation Loss={validation_loss:.6f}, LR={current_lr:.8f}")
            deep_weight_text = ", ".join([f"{target_name}={mean_deep_weights[target_index]:.4f}" for target_index, target_name in enumerate(TARGET_COLUMNS)])
            linear_weight_text = ", ".join([f"{target_name}={mean_linear_weights[target_index]:.4f}" for target_index, target_name in enumerate(TARGET_COLUMNS)])
            print(f"Deep weights: {deep_weight_text}")
            print(f"Linear weights: {linear_weight_text}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}")
            break
    model.load_state_dict(best_state)
    print(f"Best epoch={best_epoch}, best validation loss={best_validation_loss:.6f}")
    return model, pd.DataFrame(history_records), best_epoch, best_validation_loss
# ==================== 模型预测 ====================
def predict_model(model: nn.Module, data_loader: DataLoader, scaler_y: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions_scaled = []
    deep_predictions_scaled = []
    linear_predictions_scaled = []
    true_values_scaled = []
    deep_weights_all = []
    linear_weights_all = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            predictions, deep_predictions, linear_predictions, deep_weights, linear_weights = model(batch_X, return_details=True)
            predictions_scaled.append(predictions.cpu().numpy())
            deep_predictions_scaled.append(deep_predictions.cpu().numpy())
            linear_predictions_scaled.append(linear_predictions.cpu().numpy())
            true_values_scaled.append(batch_y.numpy())
            deep_weights_all.append(deep_weights.cpu().numpy())
            linear_weights_all.append(linear_weights.cpu().numpy())
    predictions_scaled = np.vstack(predictions_scaled)
    deep_predictions_scaled = np.vstack(deep_predictions_scaled)
    linear_predictions_scaled = np.vstack(linear_predictions_scaled)
    true_values_scaled = np.vstack(true_values_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    deep_predictions = scaler_y.inverse_transform(deep_predictions_scaled)
    linear_predictions = scaler_y.inverse_transform(linear_predictions_scaled)
    true_values = scaler_y.inverse_transform(true_values_scaled)
    deep_weights_all = np.vstack(deep_weights_all)
    linear_weights_all = np.vstack(linear_weights_all)
    return true_values, predictions, deep_predictions, linear_predictions, deep_weights_all, linear_weights_all
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
# ==================== Ridge初始化参数记录 ====================
def create_ridge_initialization_records(ridge_model: Ridge, fold_number: int) -> list[dict]:
    records = []
    coefficients = np.asarray(ridge_model.coef_)
    intercepts = np.asarray(ridge_model.intercept_)
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            records.append({"Fold": fold_number, "Target": target_name, "Feature": feature_name, "Ridge_Alpha": RIDGE_ALPHA, "Initial_Coefficient_Standardized": coefficients[target_index, feature_index], "Initial_Intercept_Standardized": intercepts[target_index]})
    return records
# ==================== 外层五折 + 内部独立验证集 ====================
def run_five_fold_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_all = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if y_all.ndim != 2 or y_all.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"目标矩阵必须为二维联合输出，当前形状为：{y_all.shape}")
    oof_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_deep_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_linear_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_deep_weights = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_linear_weights = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_fold = np.full(len(df), -1, dtype=int)
    metrics_records = []
    weight_records = []
    ridge_initialization_records = []
    history_frames = []
    fold_information_records = []
    outer_splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    print()
    print("=" * 80)
    print("Reduced FT-Transformer with Ridge-initialized linear branch")
    print("Outer five-fold KFold = independent test folds")
    print("Inner holdout split = training and validation")
    print("=" * 80)
    print(f"Number of samples: {len(df)}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint outputs: {TARGET_COLUMNS}")
    print(f"Number of outer folds: {N_SPLITS}")
    print(f"Inner validation ratio: {INNER_VALIDATION_RATIO:.2f}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"d_model: {D_MODEL}")
    print(f"nhead: {N_HEAD}")
    print(f"num_layers: {N_LAYERS}")
    print(f"dim_feedforward: {DIM_FEEDFORWARD}")
    print(f"dropout: {DROPOUT}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print("=" * 80)
    for fold_number, (outer_train_index, test_index) in enumerate(outer_splitter.split(X_all), start=1):
        print()
        print("=" * 80)
        print(f"Outer Fold {fold_number}/{N_SPLITS}")
        print("=" * 80)
        fold_seed = RANDOM_SEED + fold_number
        set_random_seed(fold_seed)
        inner_train_index, inner_validation_index = train_test_split(outer_train_index, test_size=INNER_VALIDATION_RATIO, shuffle=True, random_state=fold_seed)
        X_train = X_all[inner_train_index]
        y_train = y_all[inner_train_index]
        X_validation = X_all[inner_validation_index]
        y_validation = y_all[inner_validation_index]
        X_test = X_all[test_index]
        y_test = y_all[test_index]
        print(f"Outer training pool: {len(outer_train_index)}")
        print(f"Inner training samples: {len(inner_train_index)}")
        print(f"Inner validation samples: {len(inner_validation_index)}")
        print(f"Outer independent test samples: {len(test_index)}")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_validation_scaled = scaler_X.transform(X_validation)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_validation_scaled = scaler_y.transform(y_validation)
        y_test_scaled = scaler_y.transform(y_test)
        train_loader = create_data_loader(X_train_scaled, y_train_scaled, BATCH_SIZE, True, fold_seed)
        validation_loader = create_data_loader(X_validation_scaled, y_validation_scaled, BATCH_SIZE, False, fold_seed)
        test_loader = create_data_loader(X_test_scaled, y_test_scaled, BATCH_SIZE, False, fold_seed)
        model = ReducedFTTransformer(num_features=len(FEATURE_COLUMNS), num_outputs=len(TARGET_COLUMNS), d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(DEVICE)
        ridge_model = initialize_linear_branch_from_ridge(model, X_train_scaled, y_train_scaled, RIDGE_ALPHA)
        ridge_initialization_records.extend(create_ridge_initialization_records(ridge_model, fold_number))
        parameter_count = count_trainable_parameters(model)
        print(f"Trainable parameters: {parameter_count:,}")
        print("Linear branch initialized from inner-training Ridge regression.")
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            coefficient_text = ", ".join([f"{feature_name}={ridge_model.coef_[target_index, feature_index]:.6f}" for feature_index, feature_name in enumerate(FEATURE_COLUMNS)])
            print(f"{target_name} Ridge coefficients: {coefficient_text}, intercept={ridge_model.intercept_[target_index]:.6f}")
        model, fold_history, best_epoch, best_validation_loss = train_one_fold(model, train_loader, validation_loader, fold_number)
        history_frames.append(fold_history)
        y_true, y_pred, y_deep_pred, y_linear_pred, deep_weights, linear_weights = predict_model(model, test_loader, scaler_y)
        if y_pred.shape != y_test.shape:
            raise RuntimeError(f"Outer Test预测矩阵形状错误，预期{y_test.shape}，实际{y_pred.shape}")
        oof_predictions[test_index] = y_pred
        oof_deep_predictions[test_index] = y_deep_pred
        oof_linear_predictions[test_index] = y_linear_pred
        oof_deep_weights[test_index] = deep_weights
        oof_linear_weights[test_index] = linear_weights
        oof_fold[test_index] = fold_number
        fold_metrics = calculate_component_metrics(y_true, y_pred, fold_number)
        metrics_records.extend(fold_metrics)
        mean_deep_weights = deep_weights.mean(axis=0)
        std_deep_weights = deep_weights.std(axis=0, ddof=1)
        mean_linear_weights = linear_weights.mean(axis=0)
        std_linear_weights = linear_weights.std(axis=0, ddof=1)
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            weight_records.append({"Fold": fold_number, "Target": target_name, "Deep_Weight_Mean": mean_deep_weights[target_index], "Deep_Weight_Std": std_deep_weights[target_index], "Linear_Weight_Mean": mean_linear_weights[target_index], "Linear_Weight_Std": std_linear_weights[target_index]})
        fold_information_records.append({"Fold": fold_number, "Outer_Training_Pool_Samples": len(outer_train_index), "Inner_Training_Samples": len(inner_train_index), "Inner_Validation_Samples": len(inner_validation_index), "Outer_Test_Samples": len(test_index), "Best_Epoch": best_epoch, "Best_Validation_Loss": best_validation_loss, "Trainable_Parameters": parameter_count, "Ridge_Alpha": RIDGE_ALPHA, "Inner_Validation_Ratio": INNER_VALIDATION_RATIO})
        print("Outer independent test metrics:")
        for metric in fold_metrics:
            print(f"{metric['Target']}: RMSE={metric['RMSE']:.6f}, MAE={metric['MAE']:.6f}, R2={metric['R2']:.6f}, MRE={metric['MRE_percent']:.3f}%, SMAPE={metric['SMAPE_percent']:.3f}%")
        print("Fusion weights on outer independent test set:")
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            print(f"{target_name}: Deep={mean_deep_weights[target_index]:.4f} ± {std_deep_weights[target_index]:.4f}, Linear={mean_linear_weights[target_index]:.4f} ± {std_linear_weights[target_index]:.4f}")
        checkpoint = {"model_state_dict": model.state_dict(), "feature_columns": FEATURE_COLUMNS, "target_columns": TARGET_COLUMNS, "d_model": D_MODEL, "nhead": N_HEAD, "num_layers": N_LAYERS, "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT, "ridge_alpha": RIDGE_ALPHA, "ridge_initial_coef": ridge_model.coef_, "ridge_initial_intercept": ridge_model.intercept_, "fold": fold_number, "best_epoch": best_epoch, "best_validation_loss": best_validation_loss, "trainable_parameters": parameter_count, "inner_validation_ratio": INNER_VALIDATION_RATIO, "scaler_X_mean": scaler_X.mean_, "scaler_X_scale": scaler_X.scale_, "scaler_y_mean": scaler_y.mean_, "scaler_y_scale": scaler_y.scale_, "outer_train_indices": outer_train_index, "inner_train_indices": inner_train_index, "inner_validation_indices": inner_validation_index, "outer_test_indices": test_index}
        torch.save(checkpoint, f"reduced_ridge_fttransformer_fold_{fold_number}.pth")
        del model
        del ridge_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(oof_predictions).any():
        raise RuntimeError("融合输出的OOF预测中存在缺失值。")
    if np.isnan(oof_deep_predictions).any():
        raise RuntimeError("深度分支OOF预测中存在缺失值。")
    if np.isnan(oof_linear_predictions).any():
        raise RuntimeError("线性分支OOF预测中存在缺失值。")
    if np.isnan(oof_deep_weights).any():
        raise RuntimeError("深度分支融合权重中存在缺失值。")
    if np.isnan(oof_linear_weights).any():
        raise RuntimeError("线性分支融合权重中存在缺失值。")
    if np.any(oof_fold < 1):
        raise RuntimeError("存在未分配Outer Test Fold的样本。")
    metrics_df = pd.DataFrame(metrics_records)
    weights_df = pd.DataFrame(weight_records)
    ridge_initialization_df = pd.DataFrame(ridge_initialization_records)
    history_df = pd.concat(history_frames, ignore_index=True)
    fold_information_df = pd.DataFrame(fold_information_records)
    summary_df = metrics_df.groupby("Target")[["MSE", "RMSE", "MAE", "R2", "MRE_percent", "SMAPE_percent"]].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in summary_df.columns]
    summary_df = summary_df.reset_index()
    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        oof_df[f"{target_name}_True"] = y_all[:, target_index]
        oof_df[f"{target_name}_Pred"] = oof_predictions[:, target_index]
        oof_df[f"{target_name}_Deep_Pred"] = oof_deep_predictions[:, target_index]
        oof_df[f"{target_name}_Linear_Pred"] = oof_linear_predictions[:, target_index]
        oof_df[f"{target_name}_Deep_Weight"] = oof_deep_weights[:, target_index]
        oof_df[f"{target_name}_Linear_Weight"] = oof_linear_weights[:, target_index]
        oof_df[f"{target_name}_Error"] = oof_predictions[:, target_index] - y_all[:, target_index]
        oof_df[f"{target_name}_Absolute_Error"] = np.abs(oof_predictions[:, target_index] - y_all[:, target_index])
    return metrics_df, summary_df, oof_df, weights_df, ridge_initialization_df, history_df, fold_information_df
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
# ==================== 各分支OOF指标 ====================
def calculate_branch_oof_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    branch_mapping = {"Fused": "Pred", "Deep": "Deep_Pred", "Linear": "Linear_Pred"}
    for branch_name, prediction_suffix in branch_mapping.items():
        for target_name in TARGET_COLUMNS:
            y_true = oof_df[f"{target_name}_True"].to_numpy()
            y_pred = oof_df[f"{target_name}_{prediction_suffix}"].to_numpy()
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            records.append({"Branch": branch_name, "Target": target_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(records)
# ==================== OOF散点图 ====================
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
        plt.title(f"Reduced Ridge-Initialized FT-Transformer OOF Prediction for {target_name}\n$R^2$ = {r2:.4f}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"reduced_ridge_fttransformer_oof_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 五折误差图 ====================
def plot_fold_metrics(metrics_df: pd.DataFrame) -> None:
    for target_name in TARGET_COLUMNS:
        target_metrics = metrics_df.loc[metrics_df["Target"] == target_name].sort_values("Fold")
        plt.figure(figsize=(8, 6))
        plt.plot(target_metrics["Fold"], target_metrics["RMSE"], marker="o", linewidth=2, label="RMSE")
        plt.plot(target_metrics["Fold"], target_metrics["MAE"], marker="s", linewidth=2, label="MAE")
        plt.xlabel("Outer Fold", fontsize=16)
        plt.ylabel("Test Error", fontsize=16)
        plt.title(f"Outer-Test Errors for {target_name}", fontsize=17)
        plt.xticks(range(1, N_SPLITS + 1))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"reduced_ridge_fttransformer_fold_metrics_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 融合权重分布图 ====================
def plot_fusion_weight_distribution(oof_df: pd.DataFrame) -> None:
    for target_name in TARGET_COLUMNS:
        deep_weights = oof_df[f"{target_name}_Deep_Weight"].to_numpy()
        linear_weights = oof_df[f"{target_name}_Linear_Weight"].to_numpy()
        plt.figure(figsize=(8, 6))
        plt.hist(deep_weights, bins=20, alpha=0.65, label="Deep weight")
        plt.hist(linear_weights, bins=20, alpha=0.65, label="Linear weight")
        plt.xlabel("Fusion weight", fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.title(f"Outer-Test Fusion Weight Distribution for {target_name}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"reduced_ridge_fttransformer_weights_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 训练历史图 ====================
def plot_training_history(history_df: pd.DataFrame) -> None:
    for fold_number in range(1, N_SPLITS + 1):
        fold_history = history_df.loc[history_df["Fold"] == fold_number]
        if fold_history.empty:
            continue
        plt.figure(figsize=(8, 6))
        plt.plot(fold_history["Epoch"], fold_history["Train_Loss"], linewidth=2, label="Inner training loss")
        plt.plot(fold_history["Epoch"], fold_history["Validation_Loss"], linewidth=2, label="Inner validation loss")
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Standardized MSE loss", fontsize=16)
        plt.title(f"Reduced Ridge-Initialized FT-Transformer - Outer Fold {fold_number}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"reduced_ridge_fttransformer_history_fold_{fold_number}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 输出五折汇总 ====================
def print_cv_summary(summary_df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("Five-fold independent outer-test summary")
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    training_start_time = time.perf_counter()
    metrics_df, summary_df, oof_df, weights_df, ridge_initialization_df, history_df, fold_information_df = run_five_fold_cv(df)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_end_time = time.perf_counter()
    total_training_seconds = training_end_time - training_start_time
    trainable_parameters = int(fold_information_df["Trainable_Parameters"].iloc[0])
    mean_fold_seconds = total_training_seconds / N_SPLITS
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        device_name = "CPU"
        peak_gpu_memory_mb = np.nan
    computational_cost_df = pd.DataFrame([{"Model": "Ridge-initialized gated FT-Transformer", "Dataset": DATA_FILE.name, "Device": device_name, "Samples": len(df), "Number_of_Outer_Folds": N_SPLITS, "Inner_Validation_Ratio": INNER_VALIDATION_RATIO, "Trainable_Parameters": trainable_parameters, "Parameters_Million": trainable_parameters / 1_000_000, "Total_Five_Fold_Time_Seconds": total_training_seconds, "Total_Five_Fold_Time_Minutes": total_training_seconds / 60.0, "Mean_Time_Per_Fold_Seconds": mean_fold_seconds, "Mean_Time_Per_Fold_Minutes": mean_fold_seconds / 60.0, "Peak_GPU_Memory_MB": peak_gpu_memory_mb}])
    oof_metrics_df = calculate_oof_metrics(oof_df)
    branch_metrics_df = calculate_branch_oof_metrics(oof_df)
    metrics_df.to_csv("reduced_ridge_fttransformer_metrics_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_df.to_csv("reduced_ridge_fttransformer_metrics_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_df.to_csv("reduced_ridge_fttransformer_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_metrics_df.to_csv("reduced_ridge_fttransformer_oof_overall_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    branch_metrics_df.to_csv("reduced_ridge_fttransformer_branch_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    weights_df.to_csv("reduced_ridge_fttransformer_fusion_weights.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    ridge_initialization_df.to_csv("reduced_ridge_fttransformer_ridge_initialization.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    history_df.to_csv("reduced_ridge_fttransformer_training_history.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    fold_information_df.to_csv("reduced_ridge_fttransformer_fold_information.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    computational_cost_df.to_csv("reduced_ridge_fttransformer_computational_cost.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print_cv_summary(summary_df)
    print()
    print("=" * 80)
    print("Overall independent outer-test OOF metrics")
    print("=" * 80)
    print(oof_metrics_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Branch independent outer-test OOF metrics")
    print("=" * 80)
    print(branch_metrics_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Data split information")
    print("=" * 80)
    print(fold_information_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Computational cost")
    print("=" * 80)
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Model size: {trainable_parameters / 1_000_000:.4f} M parameters")
    print(f"Five-fold total training time: {total_training_seconds:.2f} s")
    print(f"Five-fold total training time: {total_training_seconds / 60.0:.2f} min")
    print(f"Mean training time per fold: {mean_fold_seconds:.2f} s")
    print(f"Mean training time per fold: {mean_fold_seconds / 60.0:.2f} min")
    print(f"Device: {device_name}")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {peak_gpu_memory_mb:.2f} MB")
    print("=" * 80)
    print()
    print("Computational cost table:")
    print(computational_cost_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)
    print("reduced_ridge_fttransformer_metrics_each_fold.csv")
    print("reduced_ridge_fttransformer_metrics_summary.csv")
    print("reduced_ridge_fttransformer_oof_predictions.csv")
    print("reduced_ridge_fttransformer_oof_overall_metrics.csv")
    print("reduced_ridge_fttransformer_branch_metrics.csv")
    print("reduced_ridge_fttransformer_fusion_weights.csv")
    print("reduced_ridge_fttransformer_ridge_initialization.csv")
    print("reduced_ridge_fttransformer_training_history.csv")
    print("reduced_ridge_fttransformer_fold_information.csv")
    print("reduced_ridge_fttransformer_computational_cost.csv")
    print("reduced_ridge_fttransformer_fold_1.pth ... fold_5.pth")
    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_fusion_weight_distribution(oof_df)
    plot_training_history(history_df)
    print()
    print("All reduced Ridge-initialized FT-Transformer tasks completed.")
if __name__ == "__main__":
    main()
