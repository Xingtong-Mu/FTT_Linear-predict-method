# %%
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os
import random
import time
import warnings
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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
LR_SCHEDULER_PATIENCE = 15

D_MODEL = 128
N_HEAD = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

FIXED_DEEP_WEIGHT = 0.5
FIXED_LINEAR_WEIGHT = 1.0 - FIXED_DEEP_WEIGHT

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


# ==================== 设备同步 ====================
def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ==================== 字段名兼容 ====================
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "MA": ["MA", "Mach", "MACH", "mach", "M0", "Mo", "MO"],
        "BETA": ["BETA", "Beta", "beta", "Sideslip", "Sideslip_deg"],
        "AA": ["AA", "Alpha_deg", "ALPHA", "Alpha", "alpha", "AOA", "AoA"],
        "CL": ["CL", "Cl", "cl", "CZI", "Czi", "czi"],
        "CM": ["CM", "Cm", "cm", "CMI25", "Cmi25", "cmi25"],
        "CD": ["CD", "Cd", "cd"],
    }

    df = df.copy()
    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    lookup = {
        str(column).casefold(): column
        for column in df.columns
    }

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
    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "cp1252",
        "latin1",
    ]

    separators = [",", "\t", ";", None]
    last_error: Exception | None = None

    for encoding in encodings:
        for separator in separators:
            try:
                if separator is None:
                    df = pd.read_csv(
                        file_path,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                    )
                else:
                    df = pd.read_csv(
                        file_path,
                        sep=separator,
                        encoding=encoding,
                    )

                if len(df.columns) > 1:
                    return df

            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        f"无法读取数据文件：{file_path}\n"
        f"最后一次错误：{last_error}"
    )


# ==================== 数据加载 ====================
def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}"
        )

    print("=" * 80)
    print("Loading aerodynamic dataset")
    print("=" * 80)
    print(f"File: {file_path.resolve()}")

    df = read_csv_robust(file_path)

    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    df = normalize_column_names(df)

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"数据缺少必要字段：{missing_columns}\n"
            f"实际字段：{list(df.columns)}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    original_rows = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    invalid_rows = original_rows - len(df)

    duplicate_mask = df.duplicated(
        subset=required_columns,
        keep="first",
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    df = (
        df.loc[~duplicate_mask]
        .copy()
        .reset_index(drop=True)
    )

    df["DATA_INDEX"] = np.arange(len(df))

    if len(df) < N_SPLITS:
        raise ValueError(
            f"有效样本数为{len(df)}，"
            f"无法进行{N_SPLITS}折交叉验证。"
        )

    if not 0.0 < INNER_VALIDATION_RATIO < 1.0:
        raise ValueError(
            "INNER_VALIDATION_RATIO必须位于0和1之间。"
        )

    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint output targets: {TARGET_COLUMNS}")
    print(f"Device: {DEVICE}")
    print(f"Outer folds: {N_SPLITS}")
    print(
        "Inner validation ratio: "
        f"{INNER_VALIDATION_RATIO:.2f}"
    )
    print(
        f"Fixed deep weight: {FIXED_DEEP_WEIGHT}"
    )
    print(
        f"Fixed linear weight: {FIXED_LINEAR_WEIGHT}"
    )
    print("=" * 80)

    return df


# ==================== ReGLU激活函数 ====================
class ReGLU(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


# ==================== 数值特征Tokenizer ====================
class NumericalFeatureTokenizer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(num_features, d_model)
        )

        self.bias = nn.Parameter(
            torch.randn(num_features, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.unsqueeze(-1)

        return (
            x * self.weight.unsqueeze(0)
            + self.bias.unsqueeze(0)
        )


# ==================== ReGLU Transformer编码层 ====================
class TransformerEncoderLayerReGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward * 2,
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )

        self.activation = ReGLU()

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.self_attn(
            src,
            src,
            src,
            need_weights=False,
        )[0]

        src = self.norm1(
            src + self.dropout1(attention_output)
        )

        feedforward_output = self.linear1(src)
        feedforward_output = self.activation(
            feedforward_output
        )
        feedforward_output = self.dropout(
            feedforward_output
        )
        feedforward_output = self.linear2(
            feedforward_output
        )

        src = self.norm2(
            src + self.dropout2(feedforward_output)
        )

        return src


# ==================== 固定权重FT-Transformer ====================
class FixedWeightFTTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        deep_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if not 0.0 <= deep_weight <= 1.0:
            raise ValueError(
                "deep_weight必须位于0和1之间。"
            )

        self.num_features = num_features
        self.num_outputs = num_outputs

        self.deep_weight = float(deep_weight)
        self.linear_weight = 1.0 - self.deep_weight

        self.numerical_tokenizer = (
            NumericalFeatureTokenizer(
                num_features,
                d_model,
            )
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_features + 1,
                d_model,
            )
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerReGLU(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_outputs),
        )

        self.linear_shortcut = nn.Linear(
            num_features,
            num_outputs,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        nn.init.zeros_(
            self.linear_shortcut.weight
        )

        nn.init.zeros_(
            self.linear_shortcut.bias
        )

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ):
        batch_size = x.size(0)

        linear_output = self.linear_shortcut(x)

        numerical_tokens = (
            self.numerical_tokenizer(x)
        )

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        deep_input = torch.cat(
            [
                cls_tokens,
                numerical_tokens,
            ],
            dim=1,
        )

        deep_input = (
            deep_input
            + self.pos_embedding
        )

        deep_input = self.dropout_layer(
            deep_input
        )

        for layer in self.transformer_layers:
            deep_input = layer(deep_input)

        cls_output = deep_input[:, 0, :]

        deep_output = self.output_head(
            cls_output
        )

        output = (
            self.deep_weight * deep_output
            + self.linear_weight * linear_output
        )

        if return_components:
            return (
                output,
                deep_output,
                linear_output,
            )

        return output


# ==================== 模型参数数量 ====================
def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==================== DataLoader ====================
def create_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    if len(dataset) == 0:
        raise ValueError(
            "DataLoader不能使用空数据集。"
        )

    actual_batch_size = min(
        batch_size,
        len(dataset),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        drop_last=False,
    )


# ==================== 单折训练 ====================
def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    fold_number: int,
) -> tuple[
    nn.Module,
    pd.DataFrame,
    int,
    float,
]:
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=1e-8,
        )
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    history_records = []

    for epoch in range(1, EPOCHS + 1):
        # -------------------- Train --------------------
        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(batch_X)
            loss = criterion(
                predictions,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            current_batch_size = batch_X.size(0)

            train_loss_sum += (
                loss.item()
                * current_batch_size
            )

            train_sample_count += (
                current_batch_size
            )

        train_loss = (
            train_loss_sum
            / train_sample_count
        )

        # -------------------- Inner Validation --------------------
        model.eval()

        validation_loss_sum = 0.0
        validation_sample_count = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(
                    DEVICE,
                    non_blocking=True,
                )

                batch_y = batch_y.to(
                    DEVICE,
                    non_blocking=True,
                )

                predictions = model(batch_X)

                loss = criterion(
                    predictions,
                    batch_y,
                )

                current_batch_size = (
                    batch_X.size(0)
                )

                validation_loss_sum += (
                    loss.item()
                    * current_batch_size
                )

                validation_sample_count += (
                    current_batch_size
                )

        validation_loss = (
            validation_loss_sum
            / validation_sample_count
        )

        scheduler.step(validation_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history_records.append(
            {
                "Fold": fold_number,
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Inner_Validation_Loss": validation_loss,
                # 保留旧列名，便于已有绘图/分析脚本兼容
                "Validation_Loss": validation_loss,
                "Learning_Rate": current_lr,
                "Fixed_Deep_Weight": FIXED_DEEP_WEIGHT,
                "Fixed_Linear_Weight": FIXED_LINEAR_WEIGHT,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS}, "
                f"Train Loss={train_loss:.6f}, "
                f"Inner Validation Loss="
                f"{validation_loss:.6f}, "
                f"LR={current_lr:.8f}"
            )

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping at epoch {epoch}, "
                f"best epoch={best_epoch}"
            )
            break

    model.load_state_dict(best_state)

    print(
        f"Best epoch={best_epoch}, "
        f"best inner validation loss="
        f"{best_validation_loss:.6f}"
    )

    return (
        model,
        pd.DataFrame(history_records),
        best_epoch,
        best_validation_loss,
    )


# ==================== 模型预测 ====================
def predict_model(
    model: nn.Module,
    data_loader: DataLoader,
    scaler_y: StandardScaler,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    predictions_scaled = []
    true_values_scaled = []
    deep_predictions_scaled = []
    linear_predictions_scaled = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True,
            )

            (
                predictions,
                deep_predictions,
                linear_predictions,
            ) = model(
                batch_X,
                return_components=True,
            )

            predictions_scaled.append(
                predictions.cpu().numpy()
            )

            deep_predictions_scaled.append(
                deep_predictions.cpu().numpy()
            )

            linear_predictions_scaled.append(
                linear_predictions.cpu().numpy()
            )

            true_values_scaled.append(
                batch_y.numpy()
            )

    predictions_scaled = np.vstack(
        predictions_scaled
    )

    deep_predictions_scaled = np.vstack(
        deep_predictions_scaled
    )

    linear_predictions_scaled = np.vstack(
        linear_predictions_scaled
    )

    true_values_scaled = np.vstack(
        true_values_scaled
    )

    predictions = scaler_y.inverse_transform(
        predictions_scaled
    )

    deep_predictions = scaler_y.inverse_transform(
        deep_predictions_scaled
    )

    linear_predictions = scaler_y.inverse_transform(
        linear_predictions_scaled
    )

    true_values = scaler_y.inverse_transform(
        true_values_scaled
    )

    return (
        true_values,
        predictions,
        deep_predictions,
        linear_predictions,
    )


# ==================== 分量指标 ====================
def calculate_component_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_number: int,
) -> list[dict]:
    records = []
    epsilon = 1e-8

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):
        true_component = y_true[
            :,
            target_index,
        ]

        predicted_component = y_pred[
            :,
            target_index,
        ]

        mse = mean_squared_error(
            true_component,
            predicted_component,
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            true_component,
            predicted_component,
        )

        r2 = r2_score(
            true_component,
            predicted_component,
        )

        mre = np.mean(
            np.abs(
                (
                    true_component
                    - predicted_component
                )
                / (
                    np.abs(true_component)
                    + epsilon
                )
            )
        ) * 100.0

        smape = np.mean(
            2.0
            * np.abs(
                predicted_component
                - true_component
            )
            / (
                np.abs(true_component)
                + np.abs(predicted_component)
                + epsilon
            )
        ) * 100.0

        records.append(
            {
                "Fold": fold_number,
                "Target": target_name,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MRE_percent": mre,
                "SMAPE_percent": smape,
            }
        )

    return records


# ==================== 外层五折 + 内部独立验证集 ====================
def run_five_fold_cv(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    X_all = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    y_all = df[
        TARGET_COLUMNS
    ].to_numpy(dtype=np.float64)

    if (
        y_all.ndim != 2
        or y_all.shape[1] != len(TARGET_COLUMNS)
    ):
        raise ValueError(
            "目标矩阵必须为二维联合输出，"
            f"当前形状为：{y_all.shape}"
        )

    # OOF数组只允许写入外层测试折预测
    oof_predictions = np.full(
        y_all.shape,
        np.nan,
        dtype=np.float64,
    )

    oof_deep_predictions = np.full(
        y_all.shape,
        np.nan,
        dtype=np.float64,
    )

    oof_linear_predictions = np.full(
        y_all.shape,
        np.nan,
        dtype=np.float64,
    )

    oof_fold = np.full(
        len(df),
        -1,
        dtype=int,
    )

    metrics_records = []
    history_frames = []
    fold_information_records = []

    outer_splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    print()
    print("=" * 80)
    print("Fixed-weight FT-Transformer ablation experiment")
    print("Evaluation: outer five-fold independent test")
    print("Model selection: inner holdout validation")
    print("=" * 80)
    print(f"Number of samples: {len(df)}")
    print(
        f"Number of input features: "
        f"{X_all.shape[1]}"
    )
    print(
        f"Number of joint outputs: "
        f"{y_all.shape[1]}"
    )
    print(f"Joint target shape: {y_all.shape}")
    print(f"Number of outer folds: {N_SPLITS}")
    print(
        "Inner validation ratio: "
        f"{INNER_VALIDATION_RATIO:.2f}"
    )
    print(f"Random seed: {RANDOM_SEED}")
    print(f"d_model: {D_MODEL}")
    print(f"nhead: {N_HEAD}")
    print(f"num_layers: {N_LAYERS}")
    print(
        f"dim_feedforward: "
        f"{DIM_FEEDFORWARD}"
    )
    print(f"dropout: {DROPOUT}")
    print(
        f"Fixed deep weight: "
        f"{FIXED_DEEP_WEIGHT}"
    )
    print(
        f"Fixed linear weight: "
        f"{FIXED_LINEAR_WEIGHT}"
    )
    print("Removed component: adaptive fusion gate")
    print(
        "Retained components: "
        "deep FT-Transformer branch "
        "and linear shortcut"
    )
    print("=" * 80)

    for fold_number, (
        outer_train_index,
        test_index,
    ) in enumerate(
        outer_splitter.split(X_all),
        start=1,
    ):
        print()
        print("=" * 80)
        print(
            f"Outer Fold "
            f"{fold_number}/{N_SPLITS}"
        )
        print("=" * 80)

        fold_seed = (
            RANDOM_SEED
            + fold_number
        )

        set_random_seed(fold_seed)

        # 外层训练池内部再划分训练集和验证集。
        # test_index不参与该划分。
        (
            inner_train_index,
            inner_validation_index,
        ) = train_test_split(
            outer_train_index,
            test_size=INNER_VALIDATION_RATIO,
            shuffle=True,
            random_state=fold_seed,
        )

        if len(inner_train_index) == 0:
            raise RuntimeError(
                f"第{fold_number}折内部训练集为空。"
            )

        if len(inner_validation_index) == 0:
            raise RuntimeError(
                f"第{fold_number}折内部验证集为空。"
            )

        X_train = X_all[
            inner_train_index
        ]

        y_train = y_all[
            inner_train_index
        ]

        X_validation = X_all[
            inner_validation_index
        ]

        y_validation = y_all[
            inner_validation_index
        ]

        X_test = X_all[
            test_index
        ]

        y_test = y_all[
            test_index
        ]

        print(
            f"Outer training pool: "
            f"{len(outer_train_index)}"
        )
        print(
            f"Inner training samples: "
            f"{len(inner_train_index)}"
        )
        print(
            f"Inner validation samples: "
            f"{len(inner_validation_index)}"
        )
        print(
            f"Outer independent test samples: "
            f"{len(test_index)}"
        )
        print(
            f"Inner training target shape: "
            f"{y_train.shape}"
        )
        print(
            f"Inner validation target shape: "
            f"{y_validation.shape}"
        )
        print(
            f"Outer test target shape: "
            f"{y_test.shape}"
        )

        # scaler严格只在inner train上拟合
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = (
            scaler_X.fit_transform(
                X_train
            )
        )

        X_validation_scaled = (
            scaler_X.transform(
                X_validation
            )
        )

        X_test_scaled = (
            scaler_X.transform(
                X_test
            )
        )

        y_train_scaled = (
            scaler_y.fit_transform(
                y_train
            )
        )

        y_validation_scaled = (
            scaler_y.transform(
                y_validation
            )
        )

        y_test_scaled = (
            scaler_y.transform(
                y_test
            )
        )

        train_loader = create_data_loader(
            X_train_scaled,
            y_train_scaled,
            BATCH_SIZE,
            True,
            fold_seed,
        )

        validation_loader = create_data_loader(
            X_validation_scaled,
            y_validation_scaled,
            BATCH_SIZE,
            False,
            fold_seed,
        )

        # 外层测试loader只在训练和模型选择全部结束后使用
        test_loader = create_data_loader(
            X_test_scaled,
            y_test_scaled,
            BATCH_SIZE,
            False,
            fold_seed,
        )

        model = FixedWeightFTTransformer(
            num_features=len(FEATURE_COLUMNS),
            num_outputs=len(TARGET_COLUMNS),
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=N_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            deep_weight=FIXED_DEEP_WEIGHT,
        ).to(DEVICE)

        parameter_count = (
            count_trainable_parameters(model)
        )

        print(
            f"Trainable parameters: "
            f"{parameter_count:,}"
        )

        synchronize_device()

        fold_training_start = (
            time.perf_counter()
        )

        (
            model,
            fold_history,
            best_epoch,
            best_validation_loss,
        ) = train_one_fold(
            model,
            train_loader,
            validation_loader,
            fold_number,
        )

        synchronize_device()

        fold_training_seconds = (
            time.perf_counter()
            - fold_training_start
        )

        history_frames.append(
            fold_history
        )

        print(
            f"Fold training time: "
            f"{fold_training_seconds:.2f} s "
            f"({fold_training_seconds / 60.0:.2f} min)"
        )

        # 第一次使用outer test：最终独立测试预测
        (
            y_true,
            y_pred,
            y_pred_deep,
            y_pred_linear,
        ) = predict_model(
            model,
            test_loader,
            scaler_y,
        )

        if y_pred.shape != y_test.shape:
            raise RuntimeError(
                "Outer Test预测矩阵形状错误，"
                f"预期{y_test.shape}，"
                f"实际{y_pred.shape}"
            )

        if not np.allclose(
            y_true,
            y_test,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "测试DataLoader返回的真实值顺序"
                "与outer test索引不一致。"
            )

        # OOF只写入outer test位置
        oof_predictions[
            test_index
        ] = y_pred

        oof_deep_predictions[
            test_index
        ] = y_pred_deep

        oof_linear_predictions[
            test_index
        ] = y_pred_linear

        oof_fold[
            test_index
        ] = fold_number

        # 每折指标仅基于outer test
        fold_metrics = (
            calculate_component_metrics(
                y_true,
                y_pred,
                fold_number,
            )
        )

        metrics_records.extend(
            fold_metrics
        )

        fold_information_records.append(
            {
                "Fold": fold_number,
                "Outer_Training_Pool_Samples": len(
                    outer_train_index
                ),
                "Inner_Training_Samples": len(
                    inner_train_index
                ),
                "Inner_Validation_Samples": len(
                    inner_validation_index
                ),
                "Outer_Test_Samples": len(
                    test_index
                ),
                # 兼容旧字段
                "Training_Samples": len(
                    inner_train_index
                ),
                "Validation_Samples": len(
                    inner_validation_index
                ),
                "Test_Samples": len(
                    test_index
                ),
                "Best_Epoch": best_epoch,
                "Best_Inner_Validation_Loss": (
                    best_validation_loss
                ),
                # 兼容旧字段
                "Best_Validation_Loss": (
                    best_validation_loss
                ),
                "Trainable_Parameters": (
                    parameter_count
                ),
                "Training_Time_Seconds": (
                    fold_training_seconds
                ),
                "Training_Time_Minutes": (
                    fold_training_seconds
                    / 60.0
                ),
                "Fixed_Deep_Weight": (
                    FIXED_DEEP_WEIGHT
                ),
                "Fixed_Linear_Weight": (
                    FIXED_LINEAR_WEIGHT
                ),
            }
        )

        print("Outer test metrics:")

        for metric in fold_metrics:
            print(
                f"{metric['Target']}: "
                f"RMSE={metric['RMSE']:.6f}, "
                f"MAE={metric['MAE']:.6f}, "
                f"R2={metric['R2']:.6f}, "
                f"MRE={metric['MRE_percent']:.3f}%, "
                f"SMAPE="
                f"{metric['SMAPE_percent']:.3f}%"
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "d_model": D_MODEL,
            "nhead": N_HEAD,
            "num_layers": N_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "fixed_deep_weight": FIXED_DEEP_WEIGHT,
            "fixed_linear_weight": FIXED_LINEAR_WEIGHT,
            "inner_validation_ratio": (
                INNER_VALIDATION_RATIO
            ),
            "fold": fold_number,
            "best_epoch": best_epoch,
            "best_inner_validation_loss": (
                best_validation_loss
            ),
            "trainable_parameters": parameter_count,
            "training_time_seconds": (
                fold_training_seconds
            ),
            "scaler_X_mean": scaler_X.mean_,
            "scaler_X_scale": scaler_X.scale_,
            "scaler_y_mean": scaler_y.mean_,
            "scaler_y_scale": scaler_y.scale_,
            "outer_train_pool_indices": (
                outer_train_index
            ),
            "inner_train_indices": (
                inner_train_index
            ),
            "inner_validation_indices": (
                inner_validation_index
            ),
            "outer_test_indices": (
                test_index
            ),
            # 兼容旧字段名；含义已明确
            "train_indices": inner_train_index,
            "validation_indices": (
                inner_validation_index
            ),
            "test_indices": test_index,
        }

        torch.save(
            checkpoint,
            (
                "fixed_weight_fttransformer_"
                f"fold_{fold_number}.pth"
            ),
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "融合输出的Outer-Test OOF预测中存在缺失值。"
        )

    if np.isnan(oof_deep_predictions).any():
        raise RuntimeError(
            "深度分支的Outer-Test OOF预测中存在缺失值。"
        )

    if np.isnan(oof_linear_predictions).any():
        raise RuntimeError(
            "线性分支的Outer-Test OOF预测中存在缺失值。"
        )

    if np.any(oof_fold < 1):
        raise RuntimeError(
            "部分样本未被分配到outer test fold。"
        )

    metrics_df = pd.DataFrame(
        metrics_records
    )

    history_df = pd.concat(
        history_frames,
        ignore_index=True,
    )

    fold_information_df = pd.DataFrame(
        fold_information_records
    )

    summary_df = (
        metrics_df.groupby("Target")[
            [
                "MSE",
                "RMSE",
                "MAE",
                "R2",
                "MRE_percent",
                "SMAPE_percent",
            ]
        ]
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in summary_df.columns
    ]

    summary_df = (
        summary_df.reset_index()
    )

    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold
    oof_df["Evaluation_Role"] = "Outer_Test_OOF"

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):
        oof_df[
            f"{target_name}_True"
        ] = y_all[:, target_index]

        oof_df[
            f"{target_name}_Pred"
        ] = oof_predictions[:, target_index]

        oof_df[
            f"{target_name}_Deep_Pred"
        ] = oof_deep_predictions[:, target_index]

        oof_df[
            f"{target_name}_Linear_Pred"
        ] = oof_linear_predictions[:, target_index]

        oof_df[
            f"{target_name}_Error"
        ] = (
            oof_predictions[:, target_index]
            - y_all[:, target_index]
        )

        oof_df[
            f"{target_name}_Absolute_Error"
        ] = np.abs(
            oof_predictions[:, target_index]
            - y_all[:, target_index]
        )

    return (
        metrics_df,
        summary_df,
        oof_df,
        history_df,
        fold_information_df,
    )


# ==================== OOF总体指标 ====================
def calculate_oof_metrics(
    oof_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    epsilon = 1e-8

    for target_name in TARGET_COLUMNS:
        y_true = oof_df[
            f"{target_name}_True"
        ].to_numpy()

        y_pred = oof_df[
            f"{target_name}_Pred"
        ].to_numpy()

        mse = mean_squared_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        mre = np.mean(
            np.abs(
                (y_true - y_pred)
                / (
                    np.abs(y_true)
                    + epsilon
                )
            )
        ) * 100.0

        smape = np.mean(
            2.0
            * np.abs(y_pred - y_true)
            / (
                np.abs(y_true)
                + np.abs(y_pred)
                + epsilon
            )
        ) * 100.0

        records.append(
            {
                "Target": target_name,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MRE_percent": mre,
                "SMAPE_percent": smape,
            }
        )

    return pd.DataFrame(records)


# ==================== 分支OOF指标 ====================
def calculate_branch_oof_metrics(
    oof_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    branch_mapping = {
        "Fused": "Pred",
        "Deep": "Deep_Pred",
        "Linear": "Linear_Pred",
    }

    for branch_name, prediction_suffix in (
        branch_mapping.items()
    ):
        for target_name in TARGET_COLUMNS:
            y_true = oof_df[
                f"{target_name}_True"
            ].to_numpy()

            y_pred = oof_df[
                (
                    f"{target_name}_"
                    f"{prediction_suffix}"
                )
            ].to_numpy()

            mse = mean_squared_error(
                y_true,
                y_pred,
            )

            rmse = np.sqrt(mse)

            mae = mean_absolute_error(
                y_true,
                y_pred,
            )

            r2 = r2_score(
                y_true,
                y_pred,
            )

            records.append(
                {
                    "Branch": branch_name,
                    "Target": target_name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                }
            )

    return pd.DataFrame(records)


# ==================== OOF真实值与预测值图 ====================
def plot_oof_true_vs_pred(
    oof_df: pd.DataFrame,
) -> None:
    for target_name in TARGET_COLUMNS:
        true_values = oof_df[
            f"{target_name}_True"
        ].to_numpy()

        predicted_values = oof_df[
            f"{target_name}_Pred"
        ].to_numpy()

        minimum_value = min(
            true_values.min(),
            predicted_values.min(),
        )

        maximum_value = max(
            true_values.max(),
            predicted_values.max(),
        )

        r2 = r2_score(
            true_values,
            predicted_values,
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            true_values,
            predicted_values,
            alpha=0.7,
            s=35,
            edgecolors="black",
            linewidths=0.4,
        )

        plt.plot(
            [minimum_value, maximum_value],
            [minimum_value, maximum_value],
            linestyle="--",
            linewidth=2,
            label="Ideal prediction",
        )

        plt.xlabel(
            f"True {target_name}",
            fontsize=16,
        )

        plt.ylabel(
            f"Predicted {target_name}",
            fontsize=16,
        )

        plt.title(
            "Fixed-weight FT-Transformer "
            f"Outer-Test OOF Prediction for {target_name}\n"
            f"$R^2$ = {r2:.4f}",
            fontsize=17,
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            "fixed_weight_fttransformer_"
            f"cv_oof_{target_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: "
            f"{output_path.resolve()}"
        )


# ==================== 每折指标图 ====================
def plot_fold_metrics(
    metrics_df: pd.DataFrame,
) -> None:
    for target_name in TARGET_COLUMNS:
        target_metrics = (
            metrics_df.loc[
                metrics_df["Target"]
                == target_name
            ]
            .sort_values("Fold")
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            target_metrics["Fold"],
            target_metrics["RMSE"],
            marker="o",
            linewidth=2,
            label="RMSE",
        )

        plt.plot(
            target_metrics["Fold"],
            target_metrics["MAE"],
            marker="s",
            linewidth=2,
            label="MAE",
        )

        plt.xlabel(
            "Outer fold",
            fontsize=16,
        )

        plt.ylabel(
            "Outer test error",
            fontsize=16,
        )

        plt.title(
            "Fixed-weight FT-Transformer "
            f"Outer-Test Errors for {target_name}",
            fontsize=17,
        )

        plt.xticks(
            range(1, N_SPLITS + 1)
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            "fixed_weight_fttransformer_"
            f"cv_fold_metrics_{target_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: "
            f"{output_path.resolve()}"
        )


# ==================== 攻角曲线图 ====================
def plot_alpha_curves(
    oof_df: pd.DataFrame,
) -> None:
    for target_name in TARGET_COLUMNS:
        alpha_values = oof_df[
            "AA"
        ].to_numpy()

        true_values = oof_df[
            f"{target_name}_True"
        ].to_numpy()

        predicted_values = oof_df[
            f"{target_name}_Pred"
        ].to_numpy()

        sort_index = np.argsort(
            alpha_values
        )

        alpha_sorted = alpha_values[
            sort_index
        ]

        true_sorted = true_values[
            sort_index
        ]

        predicted_sorted = predicted_values[
            sort_index
        ]

        plt.figure(figsize=(8, 6))

        plt.plot(
            alpha_sorted,
            true_sorted,
            linewidth=2,
            label="True value",
        )

        plt.plot(
            alpha_sorted,
            predicted_sorted,
            linestyle="--",
            linewidth=2,
            label="Predicted value",
        )

        plt.xlabel(
            "Angle of attack (deg)",
            fontsize=16,
        )

        plt.ylabel(
            target_name,
            fontsize=16,
        )

        plt.title(
            "Fixed-weight FT-Transformer: "
            f"{target_name} versus Angle of Attack",
            fontsize=17,
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            "fixed_weight_fttransformer_"
            f"alpha_curve_{target_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: "
            f"{output_path.resolve()}"
        )


# ==================== 训练历史图 ====================
def plot_training_history(
    history_df: pd.DataFrame,
) -> None:
    for fold_number in range(
        1,
        N_SPLITS + 1,
    ):
        fold_history = history_df.loc[
            history_df["Fold"]
            == fold_number
        ]

        if fold_history.empty:
            continue

        plt.figure(figsize=(8, 6))

        plt.plot(
            fold_history["Epoch"],
            fold_history["Train_Loss"],
            linewidth=2,
            label="Training loss",
        )

        plt.plot(
            fold_history["Epoch"],
            fold_history[
                "Inner_Validation_Loss"
            ],
            linewidth=2,
            label="Inner validation loss",
        )

        plt.xlabel(
            "Epoch",
            fontsize=16,
        )

        plt.ylabel(
            "Standardized MSE loss",
            fontsize=16,
        )

        plt.title(
            "Fixed-weight FT-Transformer "
            f"Training History - Outer Fold {fold_number}",
            fontsize=17,
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            "fixed_weight_fttransformer_"
            f"training_history_fold_{fold_number}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: "
            f"{output_path.resolve()}"
        )


# ==================== 打印五折汇总 ====================
def print_cv_summary(
    summary_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print(
        "Fixed-weight FT-Transformer "
        "outer-test five-fold summary"
    )
    print("=" * 80)

    for _, row in summary_df.iterrows():
        target_name = row["Target"]

        print()
        print(f"Target: {target_name}")

        print(
            f"MSE: {row['MSE_mean']:.6f} "
            f"± {row['MSE_std']:.6f}"
        )

        print(
            f"RMSE: {row['RMSE_mean']:.6f} "
            f"± {row['RMSE_std']:.6f}"
        )

        print(
            f"MAE: {row['MAE_mean']:.6f} "
            f"± {row['MAE_std']:.6f}"
        )

        print(
            f"R2: {row['R2_mean']:.6f} "
            f"± {row['R2_std']:.6f}"
        )

        print(
            f"MRE: "
            f"{row['MRE_percent_mean']:.3f}% "
            f"± "
            f"{row['MRE_percent_std']:.3f}%"
        )

        print(
            f"SMAPE: "
            f"{row['SMAPE_percent_mean']:.3f}% "
            f"± "
            f"{row['SMAPE_percent_std']:.3f}%"
        )


# ==================== 主程序 ====================
def main() -> None:
    df = load_dataset(DATA_FILE)

    (
        metrics_df,
        summary_df,
        oof_df,
        history_df,
        fold_information_df,
    ) = run_five_fold_cv(df)

    oof_metrics_df = (
        calculate_oof_metrics(oof_df)
    )

    branch_metrics_df = (
        calculate_branch_oof_metrics(
            oof_df
        )
    )

    parameter_values = (
        fold_information_df[
            "Trainable_Parameters"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(parameter_values) != 1:
        raise RuntimeError(
            "不同折的模型参数量不一致："
            f"{parameter_values.tolist()}"
        )

    trainable_parameters = int(
        parameter_values[0]
    )

    total_training_seconds = float(
        fold_information_df[
            "Training_Time_Seconds"
        ].sum()
    )

    mean_fold_training_seconds = float(
        fold_information_df[
            "Training_Time_Seconds"
        ].mean()
    )

    std_fold_training_seconds = float(
        fold_information_df[
            "Training_Time_Seconds"
        ].std(ddof=1)
    )

    minimum_fold_training_seconds = float(
        fold_information_df[
            "Training_Time_Seconds"
        ].min()
    )

    maximum_fold_training_seconds = float(
        fold_information_df[
            "Training_Time_Seconds"
        ].max()
    )

    mean_best_epoch = float(
        fold_information_df[
            "Best_Epoch"
        ].mean()
    )

    if torch.cuda.is_available():
        device_name = (
            torch.cuda.get_device_name(0)
        )
    else:
        device_name = "CPU"

    computational_cost_df = pd.DataFrame(
        [
            {
                "Model": (
                    "Fixed-weight FT-Transformer"
                ),
                "Dataset": DATA_FILE.name,
                "Device": device_name,
                "Number_of_Samples": len(df),
                "Number_of_Outer_Folds": N_SPLITS,
                "Inner_Validation_Ratio": (
                    INNER_VALIDATION_RATIO
                ),
                "Trainable_Parameters": (
                    trainable_parameters
                ),
                "Parameters_Million": (
                    trainable_parameters
                    / 1_000_000.0
                ),
                "Total_Training_Time_Seconds": (
                    total_training_seconds
                ),
                "Total_Training_Time_Minutes": (
                    total_training_seconds
                    / 60.0
                ),
                "Mean_Training_Time_Per_Fold_Seconds": (
                    mean_fold_training_seconds
                ),
                "Mean_Training_Time_Per_Fold_Minutes": (
                    mean_fold_training_seconds
                    / 60.0
                ),
                "Std_Training_Time_Per_Fold_Seconds": (
                    std_fold_training_seconds
                ),
                "Min_Training_Time_Per_Fold_Seconds": (
                    minimum_fold_training_seconds
                ),
                "Max_Training_Time_Per_Fold_Seconds": (
                    maximum_fold_training_seconds
                ),
                "Mean_Best_Epoch": (
                    mean_best_epoch
                ),
                "Fixed_Deep_Weight": (
                    FIXED_DEEP_WEIGHT
                ),
                "Fixed_Linear_Weight": (
                    FIXED_LINEAR_WEIGHT
                ),
            }
        ]
    )

    metrics_df.to_csv(
        "fixed_weight_fttransformer_cv_metrics_each_fold.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    summary_df.to_csv(
        "fixed_weight_fttransformer_cv_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    oof_df.to_csv(
        "fixed_weight_fttransformer_cv_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    history_df.to_csv(
        "fixed_weight_fttransformer_cv_training_history.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    oof_metrics_df.to_csv(
        "fixed_weight_fttransformer_cv_oof_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    branch_metrics_df.to_csv(
        "fixed_weight_fttransformer_branch_oof_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    fold_information_df.to_csv(
        "fixed_weight_fttransformer_cv_fold_information.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    computational_cost_df.to_csv(
        "fixed_weight_fttransformer_computational_cost.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print_cv_summary(summary_df)

    print()
    print("=" * 80)
    print(
        "Fixed-weight FT-Transformer "
        "overall outer-test OOF metrics"
    )
    print("=" * 80)
    print(
        oof_metrics_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("Branch outer-test OOF metrics")
    print("=" * 80)
    print(
        branch_metrics_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("Computational cost")
    print("=" * 80)
    print(f"Device: {device_name}")

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Model parameters: "
        f"{trainable_parameters / 1_000_000.0:.4f} M"
    )

    print(
        f"Total five-fold training time: "
        f"{total_training_seconds:.2f} s"
    )

    print(
        f"Total five-fold training time: "
        f"{total_training_seconds / 60.0:.2f} min"
    )

    print(
        f"Mean training time per fold: "
        f"{mean_fold_training_seconds:.2f} s"
    )

    print(
        f"Mean training time per fold: "
        f"{mean_fold_training_seconds / 60.0:.2f} min"
    )

    print(
        f"Training-time standard deviation: "
        f"{std_fold_training_seconds:.2f} s"
    )

    print(
        f"Minimum fold training time: "
        f"{minimum_fold_training_seconds:.2f} s"
    )

    print(
        f"Maximum fold training time: "
        f"{maximum_fold_training_seconds:.2f} s"
    )

    print(
        f"Mean best epoch: "
        f"{mean_best_epoch:.2f}"
    )

    print("=" * 80)

    print()
    print("Per-fold computational cost:")

    print(
        fold_information_df[
            [
                "Fold",
                "Inner_Training_Samples",
                "Inner_Validation_Samples",
                "Outer_Test_Samples",
                "Best_Epoch",
                "Trainable_Parameters",
                "Training_Time_Seconds",
                "Training_Time_Minutes",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 80)
    print("Ablation configuration")
    print("=" * 80)
    print(
        "Model: FT-Transformer "
        "with fixed branch weights"
    )
    print(
        "Retained: deep FT-Transformer branch"
    )
    print(
        "Retained: linear shortcut branch"
    )
    print("Removed: adaptive fusion gate")
    print(
        f"Deep branch weight: "
        f"{FIXED_DEEP_WEIGHT}"
    )
    print(
        f"Linear branch weight: "
        f"{FIXED_LINEAR_WEIGHT}"
    )
    print("Loss: standardized MSE")
    print(
        f"Joint outputs: "
        f"{TARGET_COLUMNS}"
    )
    print(
        "Outer test folds are not used for "
        "early stopping, scheduler, scaling, "
        "or best-model selection."
    )

    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)
    print(
        "fixed_weight_fttransformer_"
        "cv_metrics_each_fold.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "cv_metrics_summary.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "cv_oof_predictions.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "cv_oof_overall_metrics.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "branch_oof_metrics.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "cv_training_history.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "cv_fold_information.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "computational_cost.csv"
    )
    print(
        "fixed_weight_fttransformer_"
        "fold_1.pth ... fold_5.pth"
    )

    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_alpha_curves(oof_df)
    plot_training_history(history_df)

    print()
    print(
        "All fixed-weight FT-Transformer "
        "ablation tasks completed."
    )


if __name__ == "__main__":
    main()
