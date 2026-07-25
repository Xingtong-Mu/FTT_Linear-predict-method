# %%
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import random
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
# ==================== 参数设置 ====================
# DATA_FILE = Path("data_nasa.csv")
# FEATURE_COLUMNS = ["Mach", "Re", "Alpha_deg"]
# TARGET_COLUMNS = ["CL", "Cm", "CD"]
N_SPLITS = 5
INNER_VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
ALPHA_VALUE = 0.5
RIDGE_SOLVER = "auto"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
# ==================== 随机种子 ====================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_random_seed(RANDOM_SEED)
# ==================== 字段名兼容 ====================
def normalize_column_names(df):
    aliases = {"Mach": ["Mach", "M0", "Mo", "MO", "MA", "mach"], "Alpha_deg": ["Alpha_deg", "ALPHA", "Alpha", "alpha", "AA", "AOA"], "Re": ["Re", "RE", "re", "Reynolds", "ReynoldsNumber"], "CL": ["CL", "CZI", "Czi", "czi"], "Cm": ["Cm", "CM", "CMI25", "Cmi25", "cmi25"]}
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
def read_csv_robust(file_path):
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
def load_dataset(file_path):
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
        raise ValueError(f"有效样本数为{len(df)}，不足以进行{N_SPLITS}折交叉验证。")
    print(f"Valid samples: {len(df)}")
    print(f"Removed invalid rows: {invalid_rows}")
    print(f"Removed duplicate rows: {duplicate_rows}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Joint output targets: {TARGET_COLUMNS}")
    print(f"Ridge alpha: {ALPHA_VALUE}")
    print("=" * 80)
    return df
# ==================== 创建联合多输出岭回归模型 ====================
def create_ridge_model(fold_number):
    model = Ridge(alpha=ALPHA_VALUE, solver=RIDGE_SOLVER, random_state=RANDOM_SEED + fold_number)
    return model
# ==================== 分量指标计算 ====================
def calculate_component_metrics(y_true, y_pred, fold_number):
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
# ==================== 计算原始尺度回归系数 ====================
def calculate_original_scale_coefficients(model, scaler_X, scaler_y, fold_number):
    standardized_coefficients = np.asarray(model.coef_, dtype=np.float64)
    standardized_intercepts = np.asarray(model.intercept_, dtype=np.float64)
    if standardized_coefficients.ndim == 1:
        standardized_coefficients = standardized_coefficients.reshape(1, -1)
    if standardized_intercepts.ndim == 0:
        standardized_intercepts = standardized_intercepts.reshape(1)
    original_coefficients = standardized_coefficients * scaler_y.scale_[:, np.newaxis] / scaler_X.scale_[np.newaxis, :]
    original_intercepts = scaler_y.mean_ + scaler_y.scale_ * standardized_intercepts - np.sum(original_coefficients * scaler_X.mean_[np.newaxis, :], axis=1)
    records = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            records.append({"Fold": fold_number, "Target": target_name, "Feature": feature_name, "Coefficient_Standardized": standardized_coefficients[target_index, feature_index], "Coefficient_Original": original_coefficients[target_index, feature_index], "Intercept_Standardized": standardized_intercepts[target_index], "Intercept_Original": original_intercepts[target_index]})
    return records
# ==================== 五折KFold ====================
def run_five_fold_cv(df):
    X_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_all = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if y_all.ndim != 2 or y_all.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"目标矩阵必须是二维联合输出，当前形状为：{y_all.shape}")
    oof_predictions = np.full(y_all.shape, np.nan, dtype=np.float64)
    oof_fold = np.full(len(df), -1, dtype=int)
    metrics_records = []
    coefficient_records = []
    fold_overall_records = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    print()
    print("=" * 80)
    print("Ridge joint multi-output cross-validation: shuffled five-fold KFold")
    print("=" * 80)
    print(f"Number of samples: {len(df)}")
    print(f"Number of input features: {X_all.shape[1]}")
    print(f"Number of joint outputs: {y_all.shape[1]}")
    print(f"Target matrix shape: {y_all.shape}")
    print(f"Number of folds: {N_SPLITS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Alpha: {ALPHA_VALUE}")
    print("=" * 80)
    for fold_number, (outer_train_index, outer_test_index) in enumerate(splitter.split(X_all), start=1):
        print()
        print("=" * 80)
        print(f"Ridge Fold {fold_number}/{N_SPLITS}")
        print("=" * 80)
        fold_seed = RANDOM_SEED + fold_number
        set_random_seed(fold_seed)
        inner_train_index, inner_validation_index = train_test_split(outer_train_index, test_size=INNER_VALIDATION_RATIO, random_state=fold_seed, shuffle=True)
        X_train = X_all[inner_train_index]
        X_validation = X_all[inner_validation_index]
        X_test = X_all[outer_test_index]
        y_train = y_all[inner_train_index]
        y_validation = y_all[inner_validation_index]
        y_test = y_all[outer_test_index]
        print(f"Training samples: {len(inner_train_index)}")
        print(f"Validation samples: {len(inner_validation_index)}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Validation target shape: {y_validation.shape}")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_validation_scaled = scaler_X.transform(X_validation)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_validation_scaled = scaler_y.transform(y_validation)
        y_test_scaled = scaler_y.transform(y_test)
        if y_train_scaled.ndim != 2 or y_train_scaled.shape[1] != len(TARGET_COLUMNS):
            raise RuntimeError(f"联合输出目标矩阵形状错误：{y_train_scaled.shape}")
        if X_validation_scaled.shape[0] != y_validation_scaled.shape[0]:
            raise RuntimeError("内部验证集特征与目标样本数量不一致。")
        if X_test_scaled.shape[0] != y_test_scaled.shape[0]:
            raise RuntimeError("外层测试集特征与目标样本数量不一致。")
        model = create_ridge_model(fold_number)
        print("Training one Ridge model jointly for CL and Cm...")
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred_scaled = np.asarray(y_pred_scaled, dtype=np.float64)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        if y_pred_scaled.shape != y_test.shape:
            raise RuntimeError(f"预测矩阵形状错误，预期{y_test.shape}，实际{y_pred_scaled.shape}")
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        oof_predictions[outer_test_index] = y_pred
        oof_fold[outer_test_index] = fold_number
        fold_metrics = calculate_component_metrics(y_test, y_pred, fold_number)
        metrics_records.extend(fold_metrics)
        coefficient_records.extend(calculate_original_scale_coefficients(model, scaler_X, scaler_y, fold_number))
        joint_mse = mean_squared_error(y_test, y_pred)
        joint_rmse = np.sqrt(joint_mse)
        joint_mae = mean_absolute_error(y_test, y_pred)
        joint_r2 = r2_score(y_test, y_pred)
        fold_overall_records.append({"Fold": fold_number, "Training_Samples": len(inner_train_index), "Validation_Samples": len(inner_validation_index), "Joint_MSE": joint_mse, "Joint_RMSE": joint_rmse, "Joint_MAE": joint_mae, "Joint_R2": joint_r2})
        print("Fold component metrics:")
        for metric in fold_metrics:
            print(f"{metric['Target']}: RMSE={metric['RMSE']:.6f}, MAE={metric['MAE']:.6f}, R2={metric['R2']:.6f}, MRE={metric['MRE_percent']:.3f}%, SMAPE={metric['SMAPE_percent']:.3f}%")
        print(f"Joint output: RMSE={joint_rmse:.6f}, MAE={joint_mae:.6f}, R2={joint_r2:.6f}")
        checkpoint = {"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y, "feature_columns": FEATURE_COLUMNS, "target_columns": TARGET_COLUMNS, "alpha": ALPHA_VALUE, "fold": fold_number, "train_indices": inner_train_index, "validation_indices": inner_validation_index, "test_indices": outer_test_index}
        joblib.dump(checkpoint, f"ridge_fold_{fold_number}.joblib")
    if np.isnan(oof_predictions).any():
        raise RuntimeError("OOF预测中存在缺失值，请检查五折交叉验证过程。")
    metrics_df = pd.DataFrame(metrics_records)
    coefficients_df = pd.DataFrame(coefficient_records)
    fold_overall_df = pd.DataFrame(fold_overall_records)
    summary_df = metrics_df.groupby("Target")[["MSE", "RMSE", "MAE", "R2", "MRE_percent", "SMAPE_percent"]].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{statistic}" for metric, statistic in summary_df.columns]
    summary_df = summary_df.reset_index()
    coefficient_summary_df = coefficients_df.groupby(["Target", "Feature"])[["Coefficient_Standardized", "Coefficient_Original"]].agg(["mean", "std"])
    coefficient_summary_df.columns = [f"{coefficient}_{statistic}" for coefficient, statistic in coefficient_summary_df.columns]
    coefficient_summary_df = coefficient_summary_df.reset_index()
    oof_df = df.copy()
    oof_df["CV_Fold"] = oof_fold
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        oof_df[f"{target_name}_True"] = y_all[:, target_index]
        oof_df[f"{target_name}_Pred"] = oof_predictions[:, target_index]
        oof_df[f"{target_name}_Error"] = oof_predictions[:, target_index] - y_all[:, target_index]
        oof_df[f"{target_name}_Absolute_Error"] = np.abs(oof_predictions[:, target_index] - y_all[:, target_index])
    return metrics_df, summary_df, oof_df, coefficients_df, coefficient_summary_df, fold_overall_df
# ==================== OOF总体指标 ====================
def calculate_oof_metrics(oof_df):
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
# ==================== OOF真实值与预测值图 ====================
def plot_oof_true_vs_pred(oof_df):
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
        plt.title(f"Ridge Five-fold OOF Prediction for {target_name}\n$R^2$ = {r2:.4f}", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"ridge_cv_oof_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 每折指标图 ====================
def plot_fold_metrics(metrics_df):
    for target_name in TARGET_COLUMNS:
        target_metrics = metrics_df.loc[metrics_df["Target"] == target_name].sort_values("Fold")
        plt.figure(figsize=(8, 6))
        plt.plot(target_metrics["Fold"], target_metrics["RMSE"], marker="o", linewidth=2, label="RMSE")
        plt.plot(target_metrics["Fold"], target_metrics["MAE"], marker="s", linewidth=2, label="MAE")
        plt.xlabel("Fold", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title(f"Ridge Five-fold Error Metrics for {target_name}", fontsize=17)
        plt.xticks(range(1, N_SPLITS + 1))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"ridge_cv_fold_metrics_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 标准化回归系数图 ====================
def plot_coefficient_summary(coefficient_summary_df):
    for target_name in TARGET_COLUMNS:
        target_coefficients = coefficient_summary_df.loc[coefficient_summary_df["Target"] == target_name].copy()
        target_coefficients = target_coefficients.set_index("Feature").reindex(FEATURE_COLUMNS).reset_index()
        plt.figure(figsize=(8, 6))
        plt.bar(target_coefficients["Feature"], target_coefficients["Coefficient_Standardized_mean"], yerr=target_coefficients["Coefficient_Standardized_std"], capsize=5)
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Input feature", fontsize=16)
        plt.ylabel("Standardized coefficient", fontsize=16)
        plt.title(f"Ridge Feature Coefficients for {target_name}", fontsize=17)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        output_path = Path(f"ridge_coefficients_{target_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {output_path.resolve()}")
# ==================== 打印五折汇总 ====================
def print_cv_summary(summary_df):
    print()
    print("=" * 80)
    print("Ridge five-fold cross-validation summary")
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
# ==================== 打印系数汇总 ====================
def print_coefficient_summary(coefficient_summary_df):
    print()
    print("=" * 80)
    print("Ridge coefficient summary")
    print("=" * 80)
    for target_name in TARGET_COLUMNS:
        print()
        print(f"Target: {target_name}")
        target_coefficients = coefficient_summary_df.loc[coefficient_summary_df["Target"] == target_name].copy()
        target_coefficients = target_coefficients.set_index("Feature").reindex(FEATURE_COLUMNS).reset_index()
        for _, row in target_coefficients.iterrows():
            print(f"{row['Feature']}: standardized={row['Coefficient_Standardized_mean']:.6f} ± {row['Coefficient_Standardized_std']:.6f}, original={row['Coefficient_Original_mean']:.10g} ± {row['Coefficient_Original_std']:.10g}")
# ==================== 主程序 ====================
def main():
    df = load_dataset(DATA_FILE)
    metrics_df, summary_df, oof_df, coefficients_df, coefficient_summary_df, fold_overall_df = run_five_fold_cv(df)
    oof_metrics_df = calculate_oof_metrics(oof_df)
    metrics_df.to_csv("ridge_cv_metrics_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_df.to_csv("ridge_cv_metrics_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_df.to_csv("ridge_cv_oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    oof_metrics_df.to_csv("ridge_cv_oof_overall_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    coefficients_df.to_csv("ridge_cv_coefficients_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    coefficient_summary_df.to_csv("ridge_cv_coefficients_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    fold_overall_df.to_csv("ridge_cv_overall_each_fold.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    print_cv_summary(summary_df)
    print()
    print("=" * 80)
    print("Ridge overall out-of-fold metrics")
    print("=" * 80)
    print(oof_metrics_df.to_string(index=False))
    print_coefficient_summary(coefficient_summary_df)
    print()
    print("=" * 80)
    print("Output files")
    print("=" * 80)
    print("ridge_cv_metrics_each_fold.csv")
    print("ridge_cv_metrics_summary.csv")
    print("ridge_cv_oof_predictions.csv")
    print("ridge_cv_oof_overall_metrics.csv")
    print("ridge_cv_overall_each_fold.csv")
    print("ridge_cv_coefficients_each_fold.csv")
    print("ridge_cv_coefficients_summary.csv")
    print("ridge_fold_1.joblib ... ridge_fold_5.joblib")
    print("ridge_cv_oof_CL.png")
    print("ridge_cv_oof_Cm.png")
    print("ridge_cv_fold_metrics_CL.png")
    print("ridge_cv_fold_metrics_Cm.png")
    print("ridge_coefficients_CL.png")
    print("ridge_coefficients_Cm.png")
    plot_oof_true_vs_pred(oof_df)
    plot_fold_metrics(metrics_df)
    plot_coefficient_summary(coefficient_summary_df)
    print()
    print("All Ridge five-fold comparison tasks completed.")
if __name__ == "__main__":
    main()
