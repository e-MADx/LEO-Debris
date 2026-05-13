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

# Extract model + scaler from pipeline
xgb_model = pipeline.named_steps["xgb"]
scaler = pipeline.named_steps["scaler"]

# Sample a small batch
idx_sample = np.random.choice(X_train.shape[0], size=min(X_train.size, X_train.shape[0]), replace=False)
X_sample_raw = X_train[idx_sample]
X_sample = scaler.transform(X_sample_raw)   # MUST SCALE BEFORE SHAP

# Build SHAP masker (defines background distribution)
masker = shap.maskers.Independent(data=X_sample, max_samples=100)

# Build universal SHAP explainer
explainer = shap.Explainer(
    xgb_model.predict,        # function to explain
    masker=masker,            # background distribution
    algorithm="permutation"   # works for any model
)

# Compute SHAP values
shap_values = explainer(X_sample)

# Summary plot
# plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=True, cmap="viridis")
plt.tight_layout()
plt.show()
plt.savefig(f"{MODEL_DIR}/shap_summary.png", dpi=450)

# Extract model + scaler
xgb_model = pipeline.named_steps["xgb"]
scaler = pipeline.named_steps["scaler"]

# -------------------------------
# 1. Prepare sample batch for SHAP
# -------------------------------
idx_sample = np.random.choice(X_train.shape[0], size=min(X_train.size, X_train.shape[0]), replace=False)
X_sample_raw = X_train[idx_sample]
X_sample = scaler.transform(X_sample_raw)

# -------------------------------
# 2. Build masker + explainer
# -------------------------------
masker = shap.maskers.Independent(data=X_sample, max_samples=X_train.size)

explainer = shap.Explainer(
    xgb_model.predict,
    masker=masker,
    algorithm="permutation"    # safest and most stable for pipelines
)

# -------------------------------
# 3. Compute SHAP values
# -------------------------------
shap_values = explainer(X_sample)

# Convert to numpy (N_samples × N_features)
sv = shap_values.values

# =============================================================
#    PART A — SHAP DEPENDENCE PLOTS (one per feature)
# =============================================================
dep_dir = os.path.join(MODEL_DIR, "shap_dependence_plots")
os.makedirs(dep_dir, exist_ok=True)

for i, feat in enumerate(FEATURES):
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        ind=i,
        shap_values=sv,
        features=X_sample,
        feature_names=FEATURES,
        show=False,
        cmap = "viridis"
    )
    plt.title(f"SHAP Dependence: {feat}")
    plt.tight_layout()
    plt.savefig(f"{dep_dir}/dependence_{i}_{feat}.png")
    plt.show()

print("Saved SHAP dependence plots to:", dep_dir)

import seaborn as sns
import matplotlib.pyplot as plt

# PCA loadings matrix (features x components)
loadings = pca3.components_.T  # shape: (n_features, 3)

# plt.figure(figsize=(10, 6))
sns.heatmap(loadings,
            annot=True,
            fmt=".2f",
            # cmap="magma",
            xticklabels=["PC1", "PC2", "PC3"],
            yticklabels=FEATURES)

plt.title("PCA Loading Heatmap")
plt.tight_layout()
plt.savefig("PCA_loadings_heatmap.png", dpi=450)
plt.show()


# Use first 2 components for correlation circle
loadings_2d = pca3.components_[:2].T

plt.figure(figsize=(8, 8))
plt.axhline(0, color="gray", lw=0.8)
plt.axvline(0, color="gray", lw=0.8)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5)

# Plot each feature vector
for i, feature in enumerate(FEATURES):
    x, y = loadings_2d[i]
    plt.arrow(0, 0, x, y, color="red", linewidth=1.8, head_width=0.03)
    plt.text(x*1.1, y*1.1, feature, fontsize=11)

# plt.title("PCA Correlation Circle (PC1 vs PC2)")
plt.xlabel(f"PC1 (37%)")
plt.ylabel(f"PC2 (29%)")

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
# plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("PCA_correlation_circle.png", dpi=300)
plt.show()

OUT_DIR = "surrogate_model_outputs/feature_ranking"
os.makedirs(OUT_DIR, exist_ok=True)

# Sample SHAP values already computed earlier
sv = shap_values_obj.values
if isinstance(sv, list):
    sv = np.array(sv)

# SHAP importance = mean(|SHAP|) per feature
shap_importance = np.mean(np.abs(sv), axis=0)
ranking = sorted(zip(FEATURES, shap_importance), key=lambda x: x[1], reverse=True)

df_rank = pd.DataFrame(ranking, columns=["Feature", "MeanAbsSHAP"])
df_rank.to_csv(os.path.join(OUT_DIR, "feature_ranking.csv"), index=False)

colors = plt.cm.viridis(np.linspace(0, 1, len(df_rank)))  # colormap -> array of RGBA colors

# Bar plot
plt.figure(figsize=(10, 6))
plt.barh(df_rank["Feature"], df_rank["MeanAbsSHAP"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP| Impact")
plt.title("Feature Ranking (SHAP Importance)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_ranking_barplot.png"), dpi=300)
plt.show()

print("Feature ranking saved in:", OUT_DIR)


# ===========================================================
# 1) FEATURES & VARIABLES
# ===========================================================
FEATURES = [
    'AltBin', 'IncBin', 'Count', 'count_log',
    'Density', 'dens_log', 'Combined Mass',
    'mass_log', 'Volume_m3'
]

VARIABLES = ['AltBin', 'IncBin', 'Count', 'Density', 'Combined Mass']


# ===========================================================
# 2) BASELINE VALUES
# ===========================================================
baseline = {
    "AltBin": df["AltBin"].median(),
    "IncBin": df["IncBin"].median(),
    "Count": df["Count"].median(),
    "Density": df["Density"].median(),
    "Combined Mass": df["Combined Mass"].median(),
    "Volume_m3": df["Volume_m3"].median(),
}
baseline_count = baseline["Count"]


# ===========================================================
# 3) MAKE FEATURE VECTOR
# ===========================================================
def make_feature_vector(ind):
    AltBin, IncBin, Count, Density, Combined Mass = ind

    return np.array([
        AltBin,
        IncBin,
        Count,
        np.log10(Count + 1e-9),
        Density,
        np.log10(Density + 1e-12),
        Combined Mass,
        np.log10(Combined Mass + 1e-6),
        baseline["Volume_m3"]
    ])


# ===========================================================
# 4) PREDICTION FUNCTION (THIS IS WHAT WAS MISSING)
# ===========================================================
scaler = pipeline.named_steps["scaler"]
xgb_model = pipeline.named_steps["xgb"]

def predict_risk(ind):
    x = make_feature_vector(ind)
    x_scaled = scaler.transform([x])
    y_log = xgb_model.predict(x_scaled)[0]
    return 10**y_log   # convert log10 → real risk


# ===========================================================
# 5) CONSTRAINTS AND PARAMETERS
# ===========================================================
MAX_CAPACITY = 12000          # Example: change to your real system limit
MAX_RELATIVE_DECREASE = 0.5   # do not reduce Count by more than -50%
MAX_RELATIVE_INCREASE = 2.0   # do not increase Count by more than +200%
PENALTY_MULTIPLIER = 1e6       # strong penalty


# ===========================================================
# 6) EVALUATION FUNCTION
# ===========================================================
def evaluate_penalty_capacity(ind):
    """
    Objective 1: minimize predicted risk
    Objective 2: maximize available capacity (via minimizing its negative)
    With penalties if Count deviates too much from baseline
    """

    # Extract variables
    alt, inc, count, density, mass = ind

    # ---------- OBJ 1: risk ----------
    risk = predict_risk(ind)

    # ---------- OBJ 2: negative capacity (because DEAP minimizes) ----------
    available = MAX_CAPACITY - count
    neg_available = -available

    # ---------- Constraints (penalties) ----------
    rel_change = (count - baseline_count) / (baseline_count + 1e-12)
    penalty = 0.0

    if rel_change < -MAX_RELATIVE_DECREASE:
        penalty += abs(rel_change + MAX_RELATIVE_DECREASE) * PENALTY_MULTIPLIER

    if rel_change > MAX_RELATIVE_INCREASE:
        penalty += abs(rel_change - MAX_RELATIVE_INCREASE) * PENALTY_MULTIPLIER

    # Apply penalty to both objectives
    if penalty > 0:
        return (risk + penalty, neg_available + penalty)

    return (risk, neg_available)

# ===========================================================
# 7) DEAP: CREATOR & TOOLBOX
# ===========================================================
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd

# Fitness: minimize both objectives
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generators (respect dataset bounds)
bounds = {
    "AltBin": (df["AltBin"].min(), df["AltBin"].max()),
    "IncBin": (df["IncBin"].min(), df["IncBin"].max()),
    "Count": (df["Count"].min(), df["Count"].max()),
    "Density": (df["Density"].min(), df["Density"].max()),
    "Combined Mass": (df["Combined Mass"].min(), df["Combined Mass"].max()),
}

for v in VARIABLES:
    low, high = bounds[v]
    toolbox.register(f"attr_{v}", random.uniform, low, high)

# Create individual
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    [toolbox.__getattribute__(f"attr_{v}") for v in VARIABLES],
    n=1
)

# Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operators
toolbox.register("evaluate", evaluate_penalty_capacity)

toolbox.register("mate", tools.cxSimulatedBinaryBounded,
    low=[bounds[v][0] for v in VARIABLES],
    up=[bounds[v][1] for v in VARIABLES],
    eta=20)

toolbox.register("mutate", tools.mutPolynomialBounded,
    low=[bounds[v][0] for v in VARIABLES],
    up=[bounds[v][1] for v in VARIABLES],
    eta=20, indpb=0.2)

toolbox.register("select", tools.selNSGA2)



# ===========================================================
# 8) RUN NSGA-II
# ===========================================================

POP = 150
NGEN = 120

pop = toolbox.population(n=POP)
hof = tools.ParetoFront()

algorithms.eaMuPlusLambda(
    population=pop,
    toolbox=toolbox,
    mu=POP,
    lambda_=POP * 2,
    cxpb=0.85,
    mutpb=0.15,            # (0.85 + 0.15 = 1.0)
    ngen=NGEN,
    halloffame=hof,
    verbose=True
)

print("NSGA-II optimization complete!")
# ===========================================================
# 9) BUILD PARETO RESULT TABLE
# ===========================================================
rows = []
for ind in hof:
    risk, neg_cap = evaluate_penalty_capacity(ind)
    cap = -neg_cap                                  # convert back
    rows.append(list(ind) + [risk, cap])

df_pareto = pd.DataFrame(rows, columns=VARIABLES + ["PredictedRisk", "AvailableCapacity"])
df_pareto.to_csv("NSGA2_Pareto_Solutions.csv", index=False)

df_pareto.head()

plt.figure(figsize=(8,6))
plt.scatter(df_pareto["AvailableCapacity"], df_pareto["PredictedRisk"],
            c=df_pareto["AltBin"], cmap="viridis", s=60)
plt.xlabel("Available Capacity (slots remaining)")
plt.ylabel("Predicted Collision Risk")
plt.title("NSGA-II Pareto Front (Penalty + Capacity)")
plt.colorbar(label="AltBin")
plt.grid(True)
plt.show()