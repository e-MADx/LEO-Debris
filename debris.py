"""
Surrogate Model Pipeline for Space Debris Risk Prediction
---------------------------------------------------------
Predicts bin-level collision RISK = Pc × CombinedMass
using an XGBoost regression surrogate model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import xgboost as xgb
import shap
import joblib

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "Pc_and_Mass_per_bin.csv"   # Your generated input file
MODEL_DIR = "surrogate_model_outputs"
os.makedirs(MODEL_DIR, exist_ok=True)

EPS = 1e-12
RANDOM_STATE = 42

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)

required_cols = [
    "AltBin", "IncBin", "Pc", "CombinedMass",
    "Density", "Volume_m3", "Count"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# Use bin centers
df["alt_center"] = df["AltBin"] + 0.5
df["inc_center"] = df["IncBin"] + 0.5

# Log transforms for skewed features
df["count_log"]   = np.log1p(df["Count"])
df["dens_log"]    = np.log1p(df["Density"])
df["mass_log"]    = np.log1p(df["CombinedMass"])

# Risk calculation
df["Risk"] = df["Pc"] * df["CombinedMass"]
df["Risk_log"] = np.log10(df["Risk"].clip(lower=EPS))

FEATURES = [
    "alt_center",
    "inc_center",
    "Count",
    "count_log",
    "Density",
    "dens_log",
    "CombinedMass",
    "mass_log",
    "Volume_m3",
]

TARGET = "Risk_log"

X = df[FEATURES].fillna(0).values
y = df[TARGET].values

# =========================================================
# 3. TRAIN/TEST SPLIT
# =========================================================
# Stratify on Count so sparse bins are represented
count_bins = pd.qcut(df["Count"].rank(method="first"), q=5, labels=False, duplicates="drop")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index.values,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=count_bins
)

# =========================================================
# 4. NORMALIZATION + XGBOOST REGRESSOR
# =========================================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", xgb.XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=1,
        reg_lambda=1,
        random_state=RANDOM_STATE,
        tree_method="hist",
        objective="reg:squarederror"
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, f"{MODEL_DIR}/surrogate_model_xgb.joblib")

# =========================================================
# 5. EVALUATION
# =========================================================
y_pred_log = pipeline.predict(X_test)

# invert transform to raw risk
y_true_raw = 10 ** (y_test)
y_pred_raw = 10 ** (y_pred_log)

mae_log = mean_absolute_error(y_test, y_pred_log)
r2_log = r2_score(y_test, y_pred_log)

mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
r2_raw = r2_score(y_true_raw, y_pred_raw)

print("==== XGBoost Surrogate Performance ====")
print(f"MAE (log target)  : {mae_log:.4f}")
print(f"R²   (log target) : {r2_log:.4f}")
print(f"MAE (raw risk)    : {mae_raw:.4e}")
print(f"R²   (raw risk)   : {r2_raw:.4f}")

# save metrics
pd.Series({
    "mae_log": mae_log,
    "r2_log": r2_log,
    "mae_raw": mae_raw,
    "r2_raw": r2_raw
}).to_csv(f"{MODEL_DIR}/metrics.csv")

# =========================================================
# 6. SAVE PREDICTIONS
# =========================================================
df.loc[idx_test, "y_true_raw"] = y_true_raw
df.loc[idx_test, "y_pred_raw"] = y_pred_raw
df.to_csv(f"{MODEL_DIR}/surrogate_predictions.csv", index=False)


# =========================
# Train XGB (outside pipeline) and show multiple metrics
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

RANDOM_STATE = 42
MODEL_DIR = "surrogate_model_outputs"
EPS = 1e-12

# 1) scale data (we train outside pipeline to allow early stopping easily)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 2) build xgb model with multiple eval metrics (works with XGBoost 2.x)
model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.015,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_alpha=3,
    reg_lambda=3,
    tree_method="hist",
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
    eval_metric=["rmse", "mae"]   # ask for rmse and mae on log target
)

# 3) fit with early stopping (eval_set uses scaled X)
model.fit(
    X_train_s, y_train,
    eval_set=[(X_train_s, y_train), (X_test_s, y_test)],
    verbose=False
)

# save scaler + model
joblib.dump((scaler, model), f"{MODEL_DIR}/xgb_and_scaler.joblib")

# 4) get per-iteration evals_result (metrics are for the log target)
evals = model.evals_result()
train_rmse = evals["validation_0"]["rmse"]
valid_rmse = evals["validation_1"]["rmse"]
train_mae  = evals["validation_0"]["mae"]
valid_mae  = evals["validation_1"]["mae"]
iters = np.arange(1, len(train_rmse) + 1)

# 5) Plot training curves: RMSE & MAE (log target)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(iters, train_rmse, label="Training")
plt.plot(iters, valid_rmse, label="Validation")
# plt.axvline(model.best_iteration, color="r", linestyle="--", label="Best iter")
plt.xlabel("Iteration")
plt.xlim(0, 2000)
plt.ylim(0, 4)
plt.ylabel("RMSE")
plt.legend()
# plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(iters, train_mae, label="Training")
plt.plot(iters, valid_mae, label="Validation")
# plt.axvline(model.best_iteration, color="r", linestyle="--")
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.xlim(0, 2000)
plt.ylim(0, 4)
plt.legend()
# plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curves_rmse_mae.png", dpi=450)
plt.show()

# 6) Final predictions (log target) and invert to raw risk
y_pred_log = model.predict(X_test_s)               # predictions in log10(Risk)
y_true_log = y_test                                # actual log10(Risk)
y_pred_raw = 10 ** (y_pred_log.clip(min=np.log10(EPS)))  # convert back to raw risk
y_true_raw = 10 ** (y_true_log.clip(min=np.log10(EPS)))

# 7) Compute final summary metrics
mae_log = mean_absolute_error(y_true_log, y_pred_log)
rmse_log = mean_squared_error(y_true_log, y_pred_log)
r2_log = r2_score(y_true_log, y_pred_log)

mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
rmse_raw = mean_squared_error(y_true_raw, y_pred_raw)
r2_raw = r2_score(y_true_raw, y_pred_raw)

print("=== Final summary metrics ===")
print(f"Log target:   RMSE = {rmse_log:.4f}, MAE = {mae_log:.4f}, R2 = {r2_log:.4f}")
print(f"Raw target:   RMSE = {rmse_raw:.4e}, MAE = {mae_raw:.4e}, R2 = {r2_raw:.4f}")

# 8) Optional: pseudo-AUC for high-risk detection
# choose a threshold (e.g., top 10% highest true raw risk)
th = np.percentile(y_true_raw, 90)
y_true_bin = (y_true_raw >= th).astype(int)
# Only compute if there is at least one positive and one negative
try:
    auc = roc_auc_score(y_true_bin, y_pred_raw)
    print(f"Pseudo-AUC (detect top-10% risky bins): {auc:.4f}")
except Exception as e:
    print("Could not compute pseudo-AUC:", e)

# 9) Save metrics and curves
pd.Series({
    "rmse_log": rmse_log, "mae_log": mae_log, "r2_log": r2_log,
    "rmse_raw": rmse_raw, "mae_raw": mae_raw, "r2_raw": r2_raw
}).to_csv(f"{MODEL_DIR}/final_metrics.csv")

print("Done. Learning curves saved to:", f"{MODEL_DIR}/learning_curves_rmse_mae.png")
