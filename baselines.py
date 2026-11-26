"""Simple baselines to compare XGBoost against, scored on the same test set."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logistic regression is much slower than XGBoost on 11M rows, so it learns from a random
# sample of the training flights. (More rows barely help a linear model.)
LOGREG_SAMPLE = 2_000_000
ROUTE_SMOOTHING = 50  # flights of evidence before a route's own rate outweighs the overall rate


def majority_class(y_fit, n_test):
    """Always predict the most common class ("on time"): one constant score for everyone."""
    return np.full(n_test, y_fit.mean())


def route_rate(data, y, is_fit, rows):
    """Predict each flight's route's historical delay rate (training days only, smoothed)."""
    prior = y[is_fit].mean()
    stats = (data.loc[is_fit, ["ORIGIN", "DEST"]].assign(_y=y[is_fit])
             .groupby(["ORIGIN", "DEST"], observed=True)["_y"].agg(["sum", "count"]))
    rate = ((stats["sum"] + prior * ROUTE_SMOOTHING) / (stats["count"] + ROUTE_SMOOTHING)).rename("rate")
    matched = data.loc[rows, ["ORIGIN", "DEST"]].merge(rate, left_on=["ORIGIN", "DEST"],
                                                       right_index=True, how="left")
    return matched["rate"].fillna(prior).to_numpy()


def logistic_regression(X, y, is_fit, seed=42):
    """One-hot airlines/airports, standardize numbers, fill missing values with the median
    (plus a missing-value flag). Returns the fitted pipeline."""
    categorical = [c for c in X.columns if str(X[c].dtype) == "category"]
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
    ])
    model = make_pipeline(preprocess, LogisticRegression(max_iter=500, class_weight="balanced"))

    fit_rows = np.flatnonzero(is_fit.to_numpy())
    rng = np.random.default_rng(seed)
    sample = rng.choice(fit_rows, size=min(LOGREG_SAMPLE, len(fit_rows)), replace=False)
    model.fit(X.iloc[sample], y.iloc[sample])
    return model
