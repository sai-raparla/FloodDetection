import numpy as np
import pandas as pd


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import xgboost as xgb

#load data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")


target = "FloodProbability"
id_col = "id"

features = [c for c in train.columns if c not in [id_col, target]]

# --- feature engineering ---
orig_feats = features 

df = train.copy()
df_test = test.copy()

for d in (df, df_test):
    d["row_sum"] = d[orig_feats].sum(axis=1)
    d["row_mean"] = d[orig_feats].mean(axis=1)
    d["row_std"] = d[orig_feats].std(axis=1)
    d["row_max"] = d[orig_feats].max(axis=1)
    d["row_min"] = d[orig_feats].min(axis=1)

    d["cnt_ge_6"] = (d[orig_feats] >= 6).sum(axis=1)
    d["cnt_ge_7"] = (d[orig_feats] >= 7).sum(axis=1)
    d["cnt_ge_8"] = (d[orig_feats] >= 8).sum(axis=1)

    vals = d[orig_feats].values
    vals_sorted = np.sort(vals, axis=1)
    for i in range(vals_sorted.shape[1]):
        d[f"sorted_{i}"] = vals_sorted[:, i]

sorted_cols = [f"sorted_{i}" for i in range(len(orig_feats))]

new_features = orig_feats + [
    "row_sum", "row_mean", "row_std", "row_max", "row_min",
    "cnt_ge_6", "cnt_ge_7", "cnt_ge_8"
] + sorted_cols

X = df[new_features].values
X_test = df_test[new_features].values
y = df["FloodProbability"].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)


y_bins = pd.qcut(y, q=20, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []
seed = 2025
#model train
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_bins), start=1):
    print(f"\n===== Fold {fold} =====")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = xgb.XGBRegressor(
    n_estimators=1500,       
    learning_rate=0.03,    
    max_depth=8,          
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=seed,
    eval_metric="rmse",
)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
        #early_stopping_rounds=50
    )

    #validation predictions
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    fold_r2 = r2_score(y_val, val_pred)
    scores.append(fold_r2)
    print(f"Fold {fold} R2: {fold_r2:.5f}")

    #test predictions
    test_preds += model.predict(X_test) / skf.n_splits

print("\nCV R2 mean:", np.mean(scores))
print("CV R2 std: ", np.std(scores))

submission = pd.DataFrame({
    "id": test["id"],
    "FloodProbability": test_preds
})
submission.to_csv(f"submissions/submission_xgb_{seed}.csv", index=False)
print(f"Saved submission_xgb_{seed}.csv")