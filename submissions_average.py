import pandas as pd

s1 = pd.read_csv("submissions/submission_xgb_42.csv")
s2 = pd.read_csv("submissions/submission_xgb_123.csv")
s3 = pd.read_csv("submissions/submission_xgb_2025.csv")

sub = s1.copy()
sub["FloodProbability"] = (
    s1["FloodProbability"] +
    s2["FloodProbability"] +
    s3["FloodProbability"]
) / 3.0

sub.to_csv("submission_ensemble.csv", index=False)
