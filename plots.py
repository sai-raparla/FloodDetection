# plots.py
import pandas as pd
import matplotlib.pyplot as plt

def make_plots(oof_path="oof_predictions.csv"):
    oof = pd.read_csv(oof_path)
    y = oof["y_true"].values
    y_pred = oof["y_pred"].values

    # 1. Residuals
    residuals = y - y_pred

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.title("Residuals (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("charts/residual_hist.png", dpi=300)
    plt.show()

    # 2. Predicted vs True
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, s=5, alpha=0.3)

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("True FloodProbability")
    plt.ylabel("Predicted FloodProbability")
    plt.title("Predicted vs True FloodProbability")
    plt.tight_layout()
    plt.savefig("charts/pred_vs_true.png", dpi=300)
    plt.show()
