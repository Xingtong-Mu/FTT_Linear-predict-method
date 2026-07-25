# %%
# -*- coding: utf-8 -*-
from __future__ import annotations
import random
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from packaging.version import Version
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
# ==================== 参数设置 ====================
# DATA_FILE = Path("data_nasa.csv")
# FEATURE_COLUMNS = ["MA", "Re", "AA"]
# TARGET_COLUMNS = ["CL", "CM", "CD"]
N_SPLITS = 5
INNER_VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 100
MAX_DEPTH = 10
LEARNING_RATE = 0.02
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 1.0
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
N_JOBS = -1
TREE_METHOD = "hist"
OUTPUT_PREFIX = "lowspeed_xgb_joint"
SAVE_FOLD_MODELS = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
# ==================== 随机种子 ====================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
set_random_seed(RANDOM_SEED)
# ==================== XGBoost版本检查 ====================
def check_xgboost_version() -> None:
    current_version = Version(xgb.__version__)
    minimum_version = Version("2.0.0")
    print(f"XGBoost version: {current_version}")
    if current_version < minimum_version:
        raise RuntimeError(f"当前XGBoost版本为{current_version}，multi_output_tree要求XGBoost >= 2.0.0。\n请执行：pip install --upgrade xgboost")
# ==================== 字段名兼容 ====================
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "MA": ["MA", "Mach", "MACH", "mach", "M0", "Mo", "MO"],
        "BETA": ["BETA", "Beta", "beta", "Sideslip", "Sideslip_deg", "Beta_deg"],
        "AA": ["AA", "Alpha_deg", "ALPHA", "Alpha", "alpha", "AOA", "AoA", "Angle_of_attack"],
        "CL": ["CL", "Cl", "cl", "CZI", "Czi", "czi"],
        "CM": ["CM", "Cm", "cm", "CMI25", "Cmi25", "cmi25"],
        "CD": ["CD", "Cd", "cd", "C_D"]
    }
    df = df.copy()
    df.columns = [str(column).replace("\ufeff", "").replace("\r", "").replace("\n", "").strip() for column in df.columns]
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
# ==================== 数据文件读取 ====================
def read_csv_robust(file_path: Path) -> pd.DataFrame:
    with open(file_path, "rb") as binary_file:
        file_header = binary_file.read(16)
        binary_file.seek(0)
        sample_bytes = binary_file.read(4096)
    if file_header.startswith(b"PK\x03\x04"):
        print("Detected XLSX file structure; using pandas.read_excel.")
        return pd.read_excel(file_path, engine="openpyxl")
    if file_header.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        print("Detected XLS file structure; using pandas.read_excel.")
        return pd.read_excel(file_path)
    contains_null_bytes = b"\x00" in sample_bytes
    attempts = []
    if contains_null_bytes:
        attempts.extend([
            ("utf-16", "\t", "c"),
            ("utf-16", ",", "c"),
            ("utf-16", ";", "c"),
            ("utf-16-le", "\t", "c"),
            ("utf-16-le", ",", "c"),
            ("utf-16-be", "\t", "c"),
            ("utf-16-be", ",", "c"),
            ("utf-32", "\t", "c"),
            ("utf-32", ",", "c")
        ])
    attempts.extend([
        ("utf-8-sig", ",", "c"),
        ("utf-8", ",", "c"),
        ("gb18030", ",", "c"),
        ("utf-8-sig", "\t", "c"),
        ("utf-8", "\t", "c"),
        ("gb18030", "\t", "c"),
        ("utf-8-sig", ";", "c"),
        ("utf-8", ";", "c"),
        ("gb18030", ";", "c"),
        ("cp1252", ",", "c"),
        ("latin1", ",", "c")
    ])
    error_records = []
    for encoding, separator, engine in attempts:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=separator, engine=engine)
            if len(df.columns) > 1:
                print(f"Read successfully: encoding={encoding}, separator={repr(separator)}, engine={engine}")
                return df
        except Exception as exc:
            error_records.append(f"encoding={encoding}, separator={repr(separator)}: {exc}")
    automatic_encodings = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "utf-8", "gb18030", "cp1252", "latin1"]
    for encoding in automatic_encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=None, engine="python")
            if len(df.columns) > 1:
                print(f"Read successfully with automatic separator: encoding={encoding}")
                return df
        except Exception as exc:
            error_records.append(f"encoding={encoding}, separator=automatic: {exc}")
    recent_errors = "\n".join(error_records[-10:])
    raise RuntimeError(f"无法读取数据文件：{file_path}\n文件可能不是标准CSV，或者编码、分隔符异常。\n最近的读取错误：\n{recent_errors}")
# ==================== 数据加载 ====================
def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{file_path.resolve()}")
    print("=" * 80)
    print("Loading low-speed aerodynamic dataset")
    print("=" * 80)
    print(f"File: {file_path.resolve()}")
    df = read_csv_robust(file_path)
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    print(f"Original columns: {list(df.columns)}")
    df = normalize_column_names(df)
    print(f"Normalized columns: {list(df.columns)}")
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
    if len(df) < N_SPLITS:
        raise ValueError(f"有效样本数为{len(df)}，无法执行{N_SPLITS}折KFold。")
    minimum_outer_training_samples = len(df) - int(np.ceil(len(df) / N_SPLITS))
    if minimum_outer_training_samples < 2:
        raise ValueError("外折训练部分样本数不足，无法进一步划分独立验证集。")
    df["DATA_INDEX"] = np.arange(len(df), dtype=int)
    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint output targets: {TARGET_COLUMNS}")
    print("Feature ranges:")
    for column in FEATURE_COLUMNS:
        print(f"  {column}: [{df[column].min():.10g}, {df[column].max():.10g}]")
    print("Target ranges:")
    for column in TARGET_COLUMNS:
        print(f"  {column}: [{df[column].min():.10g}, {df[column].max():.10g}]")
    print("=" * 80)
    return df
# ==================== 联合多输出XGBoost模型 ====================
def create_joint_multioutput_model(fold_number: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        booster="gbtree",
        tree_method=TREE_METHOD,
        multi_strategy="multi_output_tree",
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=RANDOM_SEED + fold_number,
        n_jobs=N_JOBS,
        verbosity=0
    )
# ==================== 分量指标 ====================
def calculate_component_metrics(y_true: np.ndarray, y_pred: np.ndarray, fold_number: int | None = None) -> list[dict]:
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
        record = {
            "Target": target_name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MRE_percent": mre,
            "SMAPE_percent": smape
        }
        if fold_number is not None:
            record = {"Fold": fold_number, **record}
        records.append(record)
    return records
# ==================== 联合指标 ====================
def calculate_joint_metrics(y_true: np.ndarray, y_pred: np.ndarray, fold_number: int | None = None) -> dict:
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    record = {
        "Joint_MSE": mse,
        "Joint_RMSE": rmse,
        "Joint_MAE": mae,
        "Joint_R2": r2
    }
    if fold_number is not None:
        record = {"Fold": fold_number, **record}
    return record
# ==================== 索引划分检查 ====================
def validate_split_indices(inner_train_index: np.ndarray, inner_validation_index: np.ndarray, outer_test_index: np.ndarray, sample_count: int, fold_number: int) -> tuple[int, int, int]:
    train_validation_overlap = len(np.intersect1d(inner_train_index, inner_validation_index))
    train_test_overlap = len(np.intersect1d(inner_train_index, outer_test_index))
    validation_test_overlap = len(np.intersect1d(inner_validation_index, outer_test_index))
    if train_validation_overlap > 0:
        raise RuntimeError(f"Fold {fold_number}的内层训练集与独立验证集存在{train_validation_overlap}个重复索引。")
    if train_test_overlap > 0:
        raise RuntimeError(f"Fold {fold_number}的内层训练集与外层测试集存在{train_test_overlap}个重复索引。")
    if validation_test_overlap > 0:
        raise RuntimeError(f"Fold {fold_number}的独立验证集与外层测试集存在{validation_test_overlap}个重复索引。")
    all_indices = np.concatenate([inner_train_index, inner_validation_index, outer_test_index])
    unique_indices = np.unique(all_indices)
    if len(all_indices) != sample_count:
        raise RuntimeError(f"Fold {fold_number}划分后的总索引数量为{len(all_indices)}，应为{sample_count}。")
    if len(unique_indices) != sample_count:
        raise RuntimeError(f"Fold {fold_number}划分后仅覆盖{len(unique_indices)}个唯一样本，应覆盖{sample_count}个样本。")
    if unique_indices.min() != 0 or unique_indices.max() != sample_count - 1:
        raise RuntimeError(f"Fold {fold_number}划分索引范围异常。")
    return train_validation_overlap, train_test_overlap, validation_test_overlap
# ==================== 五折外层KFold ====================
def run_five_fold_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_all = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    expected_target_count = len(TARGET_COLUMNS)
    if y_all.ndim != 2 or y_all.shape[1] != expected_target_count:
        raise ValueError(f"目标矩阵必须是二维联合输出并包含{expected_target_count}个目标，当前形状为：{y_all.shape}")
    oof_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_fold = np.full(len(df), -1, dtype=int)
    metrics_records = []
    fold_records = []
    history_records = []
    outer_splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    global_training_ratio = (1.0 - 1.0 / N_SPLITS) * (1.0 - INNER_VALIDATION_RATIO)
    global_validation_ratio = (1.0 - 1.0 / N_SPLITS) * INNER_VALIDATION_RATIO
    global_test_ratio = 1.0 / N_SPLITS
    print()
    print("=" * 80)
    print("XGBoost joint multi-output nested five-fold cross-validation")
    print("=" * 80)
    print(f"Number of samples: {len(df)}")
    print(f"Number of input features: {X_all.shape[1]}")
    print(f"Number of joint outputs: {y_all.shape[1]}")
    print(f"Input matrix shape: {X_all.shape}")
    print(f"Target matrix shape: {y_all.shape}")
    print("Multi-output strategy: multi_output_tree")
    print(f"Joint targets: {', '.join(TARGET_COLUMNS)}")
    print(f"Number of outer folds: {N_SPLITS}")
    print(f"Inner validation ratio within outer-training fold: {INNER_VALIDATION_RATIO:.2%}")
    print(f"Approximate global inner-training ratio: {global_training_ratio:.2%}")
    print(f"Approximate global independent-validation ratio: {global_validation_ratio:.2%}")
    print(f"Approximate global outer-test ratio: {global_test_ratio:.2%}")
    print(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}")
    print(f"Maximum boosting rounds: {N_ESTIMATORS}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)
    for fold_number, (outer_train_index, outer_test_index) in enumerate(outer_splitter.split(X_all), start=1):
        print()
        print("=" * 80)
        print(f"XGBoost Outer Fold {fold_number}/{N_SPLITS}")
        print("=" * 80)
        fold_seed = RANDOM_SEED + fold_number
        set_random_seed(fold_seed)
        inner_train_index, inner_validation_index = train_test_split(
            outer_train_index,
            test_size=INNER_VALIDATION_RATIO,
            shuffle=True,
            random_state=fold_seed
        )
        inner_train_index = np.asarray(inner_train_index, dtype=int)
        inner_validation_index = np.asarray(inner_validation_index, dtype=int)
        outer_test_index = np.asarray(outer_test_index, dtype=int)
        train_validation_overlap, train_test_overlap, validation_test_overlap = validate_split_indices(
            inner_train_index,
            inner_validation_index,
            outer_test_index,
            len(df),
            fold_number
        )
        X_train = X_all[inner_train_index]
        X_validation = X_all[inner_validation_index]
        X_test = X_all[outer_test_index]
        y_train = y_all[inner_train_index]
        y_validation = y_all[inner_validation_index]
        y_test = y_all[outer_test_index]
        train_global_percent = len(inner_train_index) / len(df) * 100.0
        validation_global_percent = len(inner_validation_index) / len(df) * 100.0
        test_global_percent = len(outer_test_index) / len(df) * 100.0
        print(f"Inner training samples: {len(inner_train_index)} ({train_global_percent:.3f}% of global data)")
        print(f"Independent validation samples: {len(inner_validation_index)} ({validation_global_percent:.3f}% of global data)")
        print(f"Outer test samples: {len(outer_test_index)} ({test_global_percent:.3f}% of global data)")
        print(f"Inner training target shape: {y_train.shape}")
        print(f"Independent validation target shape: {y_validation.shape}")
        print(f"Outer test target shape: {y_test.shape}")
        print("Index overlap check: passed")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_validation_scaled = scaler_X.transform(X_validation)
        y_validation_scaled = scaler_y.transform(y_validation)
        X_test_scaled = scaler_X.transform(X_test)
        if y_train_scaled.ndim != 2 or y_train_scaled.shape[1] != expected_target_count:
            raise RuntimeError(f"联合输出训练目标形状错误：{y_train_scaled.shape}")
        if y_validation_scaled.ndim != 2 or y_validation_scaled.shape[1] != expected_target_count:
            raise RuntimeError(f"联合输出独立验证目标形状错误：{y_validation_scaled.shape}")
        model = create_joint_multioutput_model(fold_number)
        print(f"Training one joint XGBRegressor for {', '.join(TARGET_COLUMNS)}...")
        print("Early stopping uses only the independent inner validation set.")
        model.fit(
            X_train_scaled,
            y_train_scaled,
            eval_set=[(X_validation_scaled, y_validation_scaled)],
            verbose=False
        )
        evaluation_results = model.evals_result()
        validation_history = evaluation_results.get("validation_0", {}).get("rmse", [])
        for boosting_round, validation_rmse in enumerate(validation_history, start=1):
            history_records.append({
                "Fold": fold_number,
                "Boosting_Round": boosting_round,
                "Independent_Validation_RMSE": float(validation_rmse)
            })
        best_iteration_value = getattr(model, "best_iteration", None)
        best_score_value = getattr(model, "best_score", None)
        if best_iteration_value is None:
            best_iteration = int(model.get_booster().num_boosted_rounds() - 1)
        else:
            best_iteration = int(best_iteration_value)
        if best_score_value is None:
            best_validation_rmse = float(validation_history[-1]) if validation_history else np.nan
        else:
            best_validation_rmse = float(best_score_value)
        trained_boosting_rounds = int(model.get_booster().num_boosted_rounds())
        print(f"Best boosting iteration: {best_iteration + 1}")
        print(f"Best independent validation RMSE: {best_validation_rmse:.10g}")
        print(f"Actually trained boosting rounds: {trained_boosting_rounds}")
        y_pred_scaled = np.asarray(model.predict(X_test_scaled), dtype=np.float64)
        if y_pred_scaled.ndim == 1:
            raise RuntimeError("模型只返回了一维预测，说明联合多输出训练没有正确启用。")
        if y_pred_scaled.shape != y_test.shape:
            raise RuntimeError(f"外层测试预测矩阵形状错误，预期{y_test.shape}，实际{y_pred_scaled.shape}")
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        oof_predictions[outer_test_index] = y_pred
        oof_fold[outer_test_index] = fold_number
        fold_metrics = calculate_component_metrics(y_test, y_pred, fold_number)
        metrics_records.extend(fold_metrics)
        joint_record = calculate_joint_metrics(y_test, y_pred, fold_number)
        fold_records.append({
            "Fold": fold_number,
            "Inner_Training_Samples": len(inner_train_index),
            "Independent_Validation_Samples": len(inner_validation_index),
            "Outer_Test_Samples": len(outer_test_index),
            "Inner_Training_Global_Percent": train_global_percent,
            "Independent_Validation_Global_Percent": validation_global_percent,
            "Outer_Test_Global_Percent": test_global_percent,
            "Train_Validation_Overlap": train_validation_overlap,
            "Train_Test_Overlap": train_test_overlap,
            "Validation_Test_Overlap": validation_test_overlap,
            "Best_Boosting_Iteration": best_iteration + 1,
            "Best_Independent_Validation_RMSE": best_validation_rmse,
            "Actually_Trained_Boosting_Rounds": trained_boosting_rounds,
            "Joint_MSE": joint_record["Joint_MSE"],
            "Joint_RMSE": joint_record["Joint_RMSE"],
            "Joint_MAE": joint_record["Joint_MAE"],
            "Joint_R2": joint_record["Joint_R2"]
        })
        print("Outer-test component metrics:")
        for metric in fold_metrics:
            print(f"{metric['Target']}: RMSE={metric['RMSE']:.6f}, MAE={metric['MAE']:.6f}, R2={metric['R2']:.6f}, MRE={metric['MRE_percent']:.3f}%, SMAPE={metric['SMAPE_percent']:.3f}%")
        print(f"Outer-test joint output: RMSE={joint_record['Joint_RMSE']:.6f}, MAE={joint_record['Joint_MAE']:.6f}, R2={joint_record['Joint_R2']:.6f}")
        if SAVE_FOLD_MODELS:
            model_path = Path(f"{OUTPUT_PREFIX}_multioutput_fold_{fold_number}.json")
            preprocessing_path = Path(f"{OUTPUT_PREFIX}_preprocessing_fold_{fold_number}.joblib")
            split_path = Path(f"{OUTPUT_PREFIX}_split_fold_{fold_number}.npz")
            model.save_model(model_path)
            joblib.dump(
                {
                    "feature_columns": FEATURE_COLUMNS,
                    "target_columns": TARGET_COLUMNS,
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "fold": fold_number,
                    "inner_validation_ratio": INNER_VALIDATION_RATIO,
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                    "best_boosting_iteration": best_iteration + 1,
                    "best_independent_validation_rmse": best_validation_rmse,
                    "inner_training_samples": len(inner_train_index),
                    "independent_validation_samples": len(inner_validation_index),
                    "outer_test_samples": len(outer_test_index)
                },
                preprocessing_path
            )
            np.savez_compressed(
                split_path,
                inner_train_index=inner_train_index,
                inner_validation_index=inner_validation_index,
                outer_test_index=outer_test_index
            )
            print(f"Saved model: {model_path.resolve()}")
            print(f"Saved preprocessing: {preprocessing_path.resolve()}")
            print(f"Saved split indices: {split_path.resolve()}")
    if np.isnan(oof_predictions).any():
        missing_positions = np.argwhere(np.isnan(oof_predictions))
        raise RuntimeError(f"OOF预测中仍存在缺失值，缺失位置数量：{len(missing_positions)}。")
    if np.any(oof_fold < 1):
        raise RuntimeError("部分样本没有对应的外层交叉验证折编号。")
    fold_counts = pd.Series(oof_fold).value_counts().sort_index()
    if len(fold_counts) != N_SPLITS:
        raise RuntimeError(f"OOF折编号数量为{len(fold_counts)}，预期为{N_SPLITS}。")
    metrics_df = pd.DataFrame(metrics_records)
    fold_overall_df = pd.DataFrame(fold_records)
    history_df = pd.DataFrame(history_records)
    summary_df = metrics_df.groupby("Target")[["MSE", "RMSE", "MAE", "R2", "MRE_percent", "SMAPE_percent"]].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in summary_df.columns]
    summary_df = summary_df.reset_index()
    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        true_values = y_all[:, target_index]
        predicted_values = oof_predictions[:, target_index]
        oof_df[f"{target_name}_True"] = true_values
        oof_df[f"{target_name}_Pred"] = predicted_values
        oof_df[f"{target_name}_Error"] = predicted_values - true_values
        oof_df[f"{target_name}_Absolute_Error"] = np.abs(predicted_values - true_values)
    oof_component_metrics_df = pd.DataFrame(calculate_component_metrics(y_all, oof_predictions))
    oof_joint_metrics_df = pd.DataFrame([calculate_joint_metrics(y_all, oof_predictions)])
    return metrics_df, summary_df, oof_df, fold_overall_df, oof_component_metrics_df, oof_joint_metrics_df, history_df
# ==================== 真实值与预测值散点图 ====================
def plot_oof_true_vs_pred(oof_df: pd.DataFrame) -> None:
    for target_name in TARGET_COLUMNS:
        true_values = oof_df[f"{target_name}_True"].to_numpy(dtype=np.float64)
        predicted_values = oof_df[f"{target_name}_Pred"].to_numpy(dtype=np.float64)
        minimum_value = min(true_values.min(), predicted_values.min())
        maximum_value = max(true_values.max(), predicted_values.max())
        if np.isclose(minimum_value, maximum_value):
            margin = 1.0 if np.isclose(minimum_value, 0.0) else abs(minimum_value) * 0.05
            minimum_value -= margin
            maximum_value += margin
        r2 = r2_score(true_values, predicted_values)
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, alpha=0.7, s=35, edgecolors="black", linewidths=0.4)
        plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], linestyle="--", linewidth=2, label="Ideal prediction")
        plt.xlabel(f"True {target_name}", fontsize=16)
        plt.ylabel(f"Predicted {target_name}", fontsize=16)
        plt.title(f"XGBoost Joint Multi-output Outer-test OOF Prediction for {target_name}\n$R^2$ = {r2:.4f}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"{OUTPUT_PREFIX}_cv_oof_{target_name}.png")
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
        plt.xlabel("Outer fold", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title(f"XGBoost Joint Multi-output Outer-test Errors for {target_name}", fontsize=17)
        plt.xticks(range(1, N_SPLITS + 1))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"{OUTPUT_PREFIX}_cv_fold_metrics_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 独立验证集早停历史图 ====================
def plot_validation_history(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        print("No validation history is available.")
        return
    for fold_number in range(1, N_SPLITS + 1):
        fold_history = history_df.loc[history_df["Fold"] == fold_number].sort_values("Boosting_Round")
        if fold_history.empty:
            continue
        best_row_index = fold_history["Independent_Validation_RMSE"].idxmin()
        best_round = int(fold_history.loc[best_row_index, "Boosting_Round"])
        best_rmse = float(fold_history.loc[best_row_index, "Independent_Validation_RMSE"])
        plt.figure(figsize=(8, 6))
        plt.plot(fold_history["Boosting_Round"], fold_history["Independent_Validation_RMSE"], linewidth=2, label="Independent validation RMSE")
        plt.axvline(best_round, linestyle="--", linewidth=1.5, label=f"Best round: {best_round}")
        plt.xlabel("Boosting round", fontsize=16)
        plt.ylabel("Standardized RMSE", fontsize=16)
        plt.title(f"XGBoost Independent Validation History - Outer Fold {fold_number}\nBest RMSE = {best_rmse:.6f}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"{OUTPUT_PREFIX}_validation_history_fold_{fold_number}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 主程序 ====================
def main() -> None:
    check_xgboost_version()
    df = load_dataset(DATA_FILE)
    metrics_df, summary_df, oof_df, fold_overall_df, oof_component_metrics_df, oof_joint_metrics_df, history_df = run_five_fold_cv(df)
    metrics_df.to_csv(f"{OUTPUT_PREFIX}_cv_metrics_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_df.to_csv(f"{OUTPUT_PREFIX}_cv_metrics_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_df.to_csv(f"{OUTPUT_PREFIX}_cv_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_component_metrics_df.to_csv(f"{OUTPUT_PREFIX}_cv_oof_overall_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_joint_metrics_df.to_csv(f"{OUTPUT_PREFIX}_cv_oof_joint_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    fold_overall_df.to_csv(f"{OUTPUT_PREFIX}_cv_overall_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    history_df.to_csv(f"{OUTPUT_PREFIX}_cv_validation_history.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    print()
    print("=" * 80)
    print("XGBoost joint multi-output outer-test five-fold summary")
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
    print()
    print("=" * 80)
    print("Overall outer-test OOF component metrics")
    print("=" * 80)
    print(oof_component_metrics_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Overall outer-test OOF joint metrics")
    print("=" * 80)
    print(oof_joint_metrics_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Fold split and early-stopping summary")
    print("=" * 80)
    print(fold_overall_df.to_string(index=False))
    print()
    print("=" * 80)
    print("Model configuration")
    print("=" * 80)
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint outputs: {TARGET_COLUMNS}")
    print("One joint model per fold: Yes")
    print("Separate model for each target: No")
    print("Multi-output strategy: multi_output_tree")
    print("Outer cross-validation: shuffled five-fold KFold")
    print("Independent validation: 20% of each outer-training fold")
    print("Approximate global split: 64% training, 16% validation, 20% outer test")
    print(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}")
    print(f"Maximum boosting rounds: {N_ESTIMATORS}")
    print(f"Number of outer folds: {N_SPLITS}")
    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)
    print(f"{OUTPUT_PREFIX}_cv_metrics_each_fold.csv")
    print(f"{OUTPUT_PREFIX}_cv_metrics_summary.csv")
    print(f"{OUTPUT_PREFIX}_cv_oof_predictions.csv")
    print(f"{OUTPUT_PREFIX}_cv_oof_overall_metrics.csv")
    print(f"{OUTPUT_PREFIX}_cv_oof_joint_metrics.csv")
    print(f"{OUTPUT_PREFIX}_cv_overall_each_fold.csv")
    print(f"{OUTPUT_PREFIX}_cv_validation_history.csv")
    if SAVE_FOLD_MODELS:
        print(f"{OUTPUT_PREFIX}_multioutput_fold_1.json ... fold_{N_SPLITS}.json")
        print(f"{OUTPUT_PREFIX}_preprocessing_fold_1.joblib ... fold_{N_SPLITS}.joblib")
        print(f"{OUTPUT_PREFIX}_split_fold_1.npz ... fold_{N_SPLITS}.npz")
    for target_name in TARGET_COLUMNS:
        print(f"{OUTPUT_PREFIX}_cv_oof_{target_name}.png")
    for target_name in TARGET_COLUMNS:
        print(f"{OUTPUT_PREFIX}_cv_fold_metrics_{target_name}.png")
    print(f"{OUTPUT_PREFIX}_validation_history_fold_1.png ... fold_{N_SPLITS}.png")
    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_validation_history(history_df)
    print()
    print("All low-speed XGBoost nested joint multi-output tasks completed.")
if __name__ == "__main__":
    main()
