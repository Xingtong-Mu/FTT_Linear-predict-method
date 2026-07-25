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

D_MODEL = 128
N_HEAD = 8
N_LAYERS = 6
DIM_FEEDFORWARD = 512
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
def normalize_column_names(df):
    aliases = {
        "MA": ["MA", "Mach", "MACH", "mach", "M0", "Mo", "MO"],
        "BETA": ["BETA", "Beta", "beta", "Sideslip", "Sideslip_deg"],
        "AA": [
            "AA",
            "Alpha_deg",
            "ALPHA",
            "Alpha",
            "alpha",
            "AOA",
            "AoA"
        ],
        "CL": ["CL", "Cl", "cl", "CZI", "Czi", "czi"],
        "CM": ["CM", "Cm", "cm", "CMI25", "Cmi25", "cmi25"],
        "CD": ["CD", "Cd", "cd"]
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
    encodings = ["utf-8-sig", "utf-8", "gb18030", "cp1252", "latin1"]
    separators = [",", "\t", ";", None]

    last_error = None

    for encoding in encodings:
        for separator in separators:
            try:
                if separator is None:
                    df = pd.read_csv(
                        file_path,
                        sep=None,
                        engine="python",
                        encoding=encoding
                    )
                else:
                    df = pd.read_csv(
                        file_path,
                        sep=separator,
                        encoding=encoding
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
    print("Loading RAE2822 dataset")
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
            errors="coerce"
        )

    original_rows = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    invalid_rows = original_rows - len(df)

    duplicate_mask = df.duplicated(
        subset=required_columns,
        keep="first"
    )

    duplicate_rows = int(duplicate_mask.sum())

    df = (
        df.loc[~duplicate_mask]
        .copy()
        .reset_index(drop=True)
    )

    df["DATA_INDEX"] = np.arange(len(df))

    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Output targets: {TARGET_COLUMNS}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    return df


# ==================== 数值型TabTransformer ====================
class NumericalTabTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.d_model = d_model

        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model)
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        self.column_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_outputs)
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.zeros_(self.feature_bias)

        nn.init.normal_(
            self.column_embedding,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = x.unsqueeze(-1)

        x = (
            x * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )

        x = x + self.column_embedding
        x = self.dropout(x)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = self.transformer(x)

        cls_output = x[:, 0, :]
        output = self.output_head(cls_output)

        return output


# ==================== DataLoader ====================
def create_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int
) -> DataLoader:
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    actual_batch_size = min(
        batch_size,
        len(dataset)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    if shuffle:
        return DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            drop_last=False
        )

    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )


# ==================== 单折训练 ====================
def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    fold_number: int
) -> tuple[nn.Module, pd.DataFrame]:
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=1e-8
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    history_records = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            current_batch_size = batch_X.size(0)

            train_loss_sum += (
                loss.item() * current_batch_size
            )

            train_sample_count += current_batch_size

        train_loss = (
            train_loss_sum / train_sample_count
        )

        model.eval()

        validation_loss_sum = 0.0
        validation_sample_count = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(
                    DEVICE,
                    non_blocking=True
                )

                batch_y = batch_y.to(
                    DEVICE,
                    non_blocking=True
                )

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                current_batch_size = batch_X.size(0)

                validation_loss_sum += (
                    loss.item() * current_batch_size
                )

                validation_sample_count += (
                    current_batch_size
                )

        validation_loss = (
            validation_loss_sum
            / validation_sample_count
        )

        scheduler.step(validation_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history_records.append({
            "Fold": fold_number,
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Validation_Loss": validation_loss,
            "Learning_Rate": current_lr
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
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
                f"Validation Loss={validation_loss:.6f}, "
                f"LR={current_lr:.8f}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best epoch={best_epoch}"
            )
            break

    model.load_state_dict(best_state)

    print(
        f"Best epoch={best_epoch}, "
        f"best validation loss={best_validation_loss:.6f}"
    )

    return model, pd.DataFrame(history_records)


# ==================== 模型预测 ====================
def predict_model(
    model: nn.Module,
    data_loader: DataLoader,
    scaler_y: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    predictions_scaled = []
    true_values_scaled = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(batch_X)

            predictions_scaled.append(
                predictions.cpu().numpy()
            )

            true_values_scaled.append(
                batch_y.numpy()
            )

    predictions_scaled = np.vstack(
        predictions_scaled
    )

    true_values_scaled = np.vstack(
        true_values_scaled
    )

    predictions = scaler_y.inverse_transform(
        predictions_scaled
    )

    true_values = scaler_y.inverse_transform(
        true_values_scaled
    )

    return true_values, predictions


# ==================== 分量指标 ====================
def calculate_component_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_number: int
) -> list[dict]:
    records = []
    epsilon = 1e-8

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):
        true_component = y_true[:, target_index]
        predicted_component = y_pred[:, target_index]

        mse = mean_squared_error(
            true_component,
            predicted_component
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            true_component,
            predicted_component
        )

        r2 = r2_score(
            true_component,
            predicted_component
        )

        mre = np.mean(
            np.abs(
                (true_component - predicted_component)
                / (np.abs(true_component) + epsilon)
            )
        ) * 100.0

        smape = np.mean(
            2.0
            * np.abs(
                predicted_component - true_component
            )
            / (
                np.abs(true_component)
                + np.abs(predicted_component)
                + epsilon
            )
        ) * 100.0

        records.append({
            "Fold": fold_number,
            "Target": target_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MRE_percent": mre,
            "SMAPE_percent": smape
        })

    return records


# ==================== 五折KFold ====================
def run_five_fold_cv(
    df: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    X_all = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    y_all = df[TARGET_COLUMNS].to_numpy(
        dtype=np.float64
    )

    if (
        y_all.ndim != 2
        or y_all.shape[1] != len(TARGET_COLUMNS)
    ):
        raise ValueError(
            f"目标矩阵形状错误：{y_all.shape}"
        )

    oof_predictions = np.full(
        y_all.shape,
        np.nan,
        dtype=np.float64
    )

    oof_fold = np.full(
        len(df),
        -1,
        dtype=int
    )

    metrics_records = []
    history_frames = []

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    print()
    print("=" * 80)
    print(
        "Numerical TabTransformer cross-validation: "
        "shuffled five-fold KFold"
    )
    print(f"Number of samples: {len(df)}")
    print(f"Number of folds: {N_SPLITS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Joint target shape: {y_all.shape}")
    print(f"d_model: {D_MODEL}")
    print(f"nhead: {N_HEAD}")
    print(f"num_layers: {N_LAYERS}")
    print(f"dim_feedforward: {DIM_FEEDFORWARD}")
    print("=" * 80)

    for fold_number, (
        outer_train_index,
        outer_test_index
    ) in enumerate(
        splitter.split(X_all),
        start=1
    ):
        print()
        print("=" * 80)
        print(
            f"Numerical TabTransformer Fold "
            f"{fold_number}/{N_SPLITS}"
        )
        print("=" * 80)

        fold_seed = RANDOM_SEED + fold_number
        set_random_seed(fold_seed)

        (
            inner_train_index,
            inner_validation_index
        ) = train_test_split(
            outer_train_index,
            test_size=INNER_VALIDATION_RATIO,
            random_state=fold_seed,
            shuffle=True
        )

        X_train = X_all[inner_train_index]
        X_validation = X_all[inner_validation_index]
        X_test = X_all[outer_test_index]

        y_train = y_all[inner_train_index]
        y_validation = y_all[inner_validation_index]
        y_test = y_all[outer_test_index]

        print(f"Training samples: {len(inner_train_index)}")
        print(
            f"Validation samples: "
            f"{len(inner_validation_index)}"
        )
        print(f"Training target shape: {y_train.shape}")
        print(
            f"Validation target shape: "
            f"{y_validation.shape}"
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(
            X_train
        )

        X_validation_scaled = scaler_X.transform(
            X_validation
        )

        X_test_scaled = scaler_X.transform(
            X_test
        )

        y_train_scaled = scaler_y.fit_transform(
            y_train
        )

        y_validation_scaled = scaler_y.transform(
            y_validation
        )

        y_test_scaled = scaler_y.transform(
            y_test
        )

        train_loader = create_data_loader(
            X_train_scaled,
            y_train_scaled,
            BATCH_SIZE,
            True,
            fold_seed
        )

        validation_loader = create_data_loader(
            X_validation_scaled,
            y_validation_scaled,
            BATCH_SIZE,
            False,
            fold_seed
        )

        test_loader = create_data_loader(
            X_test_scaled,
            y_test_scaled,
            BATCH_SIZE,
            False,
            fold_seed
        )

        model = NumericalTabTransformer(
            num_features=len(FEATURE_COLUMNS),
            num_outputs=len(TARGET_COLUMNS),
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=N_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT
        ).to(DEVICE)

        model, fold_history = train_one_fold(
            model,
            train_loader,
            validation_loader,
            fold_number
        )

        history_frames.append(fold_history)

        y_true, y_pred = predict_model(
            model,
            test_loader,
            scaler_y
        )

        if y_pred.shape != y_test.shape:
            raise RuntimeError(
                f"预测矩阵形状错误，"
                f"预期{y_test.shape}，"
                f"实际{y_pred.shape}"
            )

        oof_predictions[outer_test_index] = y_pred
        oof_fold[outer_test_index] = fold_number

        fold_metrics = calculate_component_metrics(
            y_true,
            y_pred,
            fold_number
        )

        metrics_records.extend(fold_metrics)

        print("Fold metrics:")

        for metric in fold_metrics:
            print(
                f"{metric['Target']}: "
                f"RMSE={metric['RMSE']:.6f}, "
                f"MAE={metric['MAE']:.6f}, "
                f"R2={metric['R2']:.6f}, "
                f"MRE={metric['MRE_percent']:.3f}%, "
                f"SMAPE={metric['SMAPE_percent']:.3f}%"
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
            "fold": fold_number,
            "scaler_X_mean": scaler_X.mean_,
            "scaler_X_scale": scaler_X.scale_,
            "scaler_y_mean": scaler_y.mean_,
            "scaler_y_scale": scaler_y.scale_
        }

        torch.save(
            checkpoint,
            f"best_tabtransformer_fold_{fold_number}.pth"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            "OOF预测中存在缺失值，"
            "请检查五折交叉验证过程。"
        )

    metrics_df = pd.DataFrame(
        metrics_records
    )

    history_df = pd.concat(
        history_frames,
        ignore_index=True
    )

    summary_df = (
        metrics_df
        .groupby("Target")[
            [
                "MSE",
                "RMSE",
                "MAE",
                "R2",
                "MRE_percent",
                "SMAPE_percent"
            ]
        ]
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold

    for target_index, target_name in enumerate(
        TARGET_COLUMNS
    ):
        oof_df[f"{target_name}_True"] = (
            y_all[:, target_index]
        )

        oof_df[f"{target_name}_Pred"] = (
            oof_predictions[:, target_index]
        )

        oof_df[f"{target_name}_Error"] = (
            oof_predictions[:, target_index]
            - y_all[:, target_index]
        )

        oof_df[f"{target_name}_Absolute_Error"] = (
            np.abs(
                oof_predictions[:, target_index]
                - y_all[:, target_index]
            )
        )

    return (
        metrics_df,
        summary_df,
        oof_df,
        history_df
    )


# ==================== OOF总体指标 ====================
def calculate_oof_metrics(
    oof_df: pd.DataFrame
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
            y_pred
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        mre = np.mean(
            np.abs(
                (y_true - y_pred)
                / (np.abs(y_true) + epsilon)
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

        records.append({
            "Target": target_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MRE_percent": mre,
            "SMAPE_percent": smape
        })

    return pd.DataFrame(records)


# ==================== OOF真实值与预测值 ====================
def plot_oof_true_vs_pred(
    oof_df: pd.DataFrame
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
            predicted_values.min()
        )

        maximum_value = max(
            true_values.max(),
            predicted_values.max()
        )

        r2 = r2_score(
            true_values,
            predicted_values
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            true_values,
            predicted_values,
            alpha=0.7,
            s=35,
            edgecolors="black",
            linewidths=0.4
        )

        plt.plot(
            [minimum_value, maximum_value],
            [minimum_value, maximum_value],
            linestyle="--",
            linewidth=2,
            label="Ideal prediction"
        )

        plt.xlabel(
            f"True {target_name}",
            fontsize=16
        )

        plt.ylabel(
            f"Predicted {target_name}",
            fontsize=16
        )

        plt.title(
            f"Numerical TabTransformer OOF Prediction "
            f"for {target_name}\n"
            f"$R^2$ = {r2:.4f}",
            fontsize=17
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            f"tabtransformer_cv_oof_{target_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: {output_path.resolve()}"
        )


# ==================== 每折指标图 ====================
def plot_fold_metrics(
    metrics_df: pd.DataFrame
) -> None:
    for target_name in TARGET_COLUMNS:
        target_metrics = (
            metrics_df
            .loc[metrics_df["Target"] == target_name]
            .sort_values("Fold")
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            target_metrics["Fold"],
            target_metrics["RMSE"],
            marker="o",
            linewidth=2,
            label="RMSE"
        )

        plt.plot(
            target_metrics["Fold"],
            target_metrics["MAE"],
            marker="s",
            linewidth=2,
            label="MAE"
        )

        plt.xlabel(
            "Fold",
            fontsize=16
        )

        plt.ylabel(
            "Error",
            fontsize=16
        )

        plt.title(
            f"Numerical TabTransformer Five-fold "
            f"Errors for {target_name}",
            fontsize=17
        )

        plt.xticks(
            range(1, N_SPLITS + 1)
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            f"tabtransformer_cv_fold_metrics_"
            f"{target_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: {output_path.resolve()}"
        )


# ==================== 训练历史图 ====================
def plot_training_history(
    history_df: pd.DataFrame
) -> None:
    for fold_number in range(
        1,
        N_SPLITS + 1
    ):
        fold_history = history_df.loc[
            history_df["Fold"] == fold_number
        ]

        if fold_history.empty:
            continue

        plt.figure(figsize=(8, 6))

        plt.plot(
            fold_history["Epoch"],
            fold_history["Train_Loss"],
            linewidth=2,
            label="Training loss"
        )

        plt.plot(
            fold_history["Epoch"],
            fold_history["Validation_Loss"],
            linewidth=2,
            label="Validation loss"
        )

        plt.xlabel(
            "Epoch",
            fontsize=16
        )

        plt.ylabel(
            "Standardized MSE loss",
            fontsize=16
        )

        plt.title(
            f"Numerical TabTransformer Training "
            f"History - Fold {fold_number}",
            fontsize=17
        )

        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(
            f"tabtransformer_training_history_"
            f"fold_{fold_number}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(
            f"Saved plot: {output_path.resolve()}"
        )


# ==================== 打印五折汇总 ====================
def print_cv_summary(
    summary_df: pd.DataFrame
) -> None:
    print()
    print("=" * 80)
    print(
        "Numerical TabTransformer five-fold "
        "cross-validation summary"
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
            f"± {row['MRE_percent_std']:.3f}%"
        )

        print(
            f"SMAPE: "
            f"{row['SMAPE_percent_mean']:.3f}% "
            f"± {row['SMAPE_percent_std']:.3f}%"
        )


# ==================== 主程序 ====================
def main() -> None:
    df = load_dataset(DATA_FILE)

    (
        metrics_df,
        summary_df,
        oof_df,
        history_df
    ) = run_five_fold_cv(df)

    oof_metrics_df = calculate_oof_metrics(
        oof_df
    )

    metrics_df.to_csv(
        "tabtransformer_cv_metrics_each_fold.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    summary_df.to_csv(
        "tabtransformer_cv_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    oof_df.to_csv(
        "tabtransformer_cv_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    history_df.to_csv(
        "tabtransformer_cv_training_history.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    oof_metrics_df.to_csv(
        "tabtransformer_cv_oof_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    print_cv_summary(summary_df)

    print()
    print("=" * 80)
    print(
        "Numerical TabTransformer overall "
        "out-of-fold metrics"
    )
    print("=" * 80)

    print(
        oof_metrics_df.to_string(index=False)
    )

    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)

    print(
        "tabtransformer_cv_metrics_each_fold.csv"
    )
    print(
        "tabtransformer_cv_metrics_summary.csv"
    )
    print(
        "tabtransformer_cv_oof_predictions.csv"
    )
    print(
        "tabtransformer_cv_oof_overall_metrics.csv"
    )
    print(
        "tabtransformer_cv_training_history.csv"
    )
    print(
        "best_tabtransformer_fold_1.pth ... "
        "best_tabtransformer_fold_5.pth"
    )
    print(
        "tabtransformer_cv_oof_CL.png"
    )
    print(
        "tabtransformer_cv_oof_Cm.png"
    )
    print(
        "tabtransformer_cv_fold_metrics_CL.png"
    )
    print(
        "tabtransformer_cv_fold_metrics_Cm.png"
    )
    print(
        "tabtransformer_training_history_fold_1.png "
        "... fold_5.png"
    )

    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_training_history(history_df)

    print()
    print(
        "All Numerical TabTransformer "
        "five-fold comparison tasks completed."
    )


if __name__ == "__main__":
    main()
