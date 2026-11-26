import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, precision_recall_curve, recall_score,
                             roc_auc_score, average_precision_score)
from xgboost import XGBClassifier

import baselines
from features import prepare_features, ROTATION_FEATURES
from weather import WEATHER_FEATURES

MODELS_DIR = Path(__file__).parent / "models"

# -- Two models for the app --
# "advance": everything known when booking (schedule, calendar, weather forecast)
# "day_of":  adds what's known on the day of travel, ~1 hour before departure
#            (the plane's previous flight, recent delays at both airports)
ADVANCE_FEATURES = ["AIRLINE", "ORIGIN", "DEST", "DISTANCE", "DAY_OF_WEEK", "MONTH",
                    "DEP_MINUTES", "ARR_MINUTES", "DAYS_TO_HOLIDAY", "SCHEDULE_PADDING",
                    "ORIGIN_HOUR_DEPS", "ORIGIN_DAY_DEPS", "DEST_HOUR_ARRS"] + WEATHER_FEATURES
DAY_OF_FEATURES = ADVANCE_FEATURES + ROTATION_FEATURES + [
    "ORIGIN_RECENT_DEP_DELAY", "ORIGIN_RECENT_DEPS", "DEST_RECENT_ARR_DELAY"]
MODELS = {"advance": ADVANCE_FEATURES, "day_of": DAY_OF_FEATURES}

# Share of delayed flights the model should catch. Higher = more delays caught,
# but more on-time flights wrongly flagged as delayed.
TARGET_DELAY_RECALL = 0.70


def main():
    flights = prepare_features()

    # Drop cancelled/diverted flights (no arrival delay recorded)
    data = flights.dropna(subset=["ARRIVAL_DELAY"])
    del flights  # free the full copy (21M rows)
    # A flight counts as delayed if it arrives 15+ minutes late (the official definition)
    data["delayed"] = (data["ARRIVAL_DELAY"] >= 15).astype(int)

    # Split by time: learn from 2023-2024, test on all of 2025 (like predicting the future).
    # Validation = random 20% of 2023-2024 days, used for early stopping and the threshold.
    # Whole days go to one side, so a storm day can't appear in both.
    y = data["delayed"]
    is_test = data["YEAR"] == 2025
    train_days = data.loc[~is_test, "DAY_INDEX"].unique()
    val_days = np.random.default_rng(42).permutation(train_days)[:int(len(train_days) * 0.2)]
    is_val = data["DAY_INDEX"].isin(val_days) & ~is_test
    is_fit = ~is_test & ~is_val
    print(f"Fit {is_fit.sum():,} / validation {is_val.sum():,} / test {is_test.sum():,} flights")

    MODELS_DIR.mkdir(exist_ok=True)
    y_test = y[is_test]
    results = []  # one row per model for the comparison table at the end
    for name, features in MODELS.items():
        print(f"\n=== {name} model ({len(features)} features) ===")
        model, threshold = train_model(data[features], y, is_fit, is_val)

        probs = model.predict_proba(data.loc[is_test, features])[:, 1]
        preds = (probs >= threshold).astype(int)
        results.append(score(f"XGBoost ({name})", y_test, probs, threshold))
        print(classification_report(y_test, preds, target_names=["on time", "delayed"]))
        print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
        print("Avg precision:", round(average_precision_score(y_test, probs), 4))
        importances = pd.Series(model.feature_importances_, index=features)
        print(importances.sort_values(ascending=False).head(15).round(3).to_string())

        save_model(name, model, features, threshold, data)
        if "--plot" in sys.argv:
            from visuals import plot_results
            plot_results(y_test, preds, probs)

    if "--skip-baselines" not in sys.argv:
        results += run_baselines(data, y, is_fit, is_val, is_test)
    print("\n=== Comparison on the 2025 test set ===")
    print(pd.DataFrame(results).to_string(index=False))


def run_baselines(data, y, is_fit, is_val, is_test):
    """Same test set and same threshold rule (70% delayed recall on validation) as XGBoost."""
    y_val, y_test = y[is_val], y[is_test]
    rows = []

    print("\n=== Baseline: majority class (always 'on time') ===")
    constant = baselines.majority_class(y[is_fit], is_test.sum())
    rows.append(score("Always 'on time'", y_test, constant, threshold=1.0))

    print("=== Baseline: route's historical delay rate ===")
    val_rate = baselines.route_rate(data, y, is_fit, is_val)
    test_rate = baselines.route_rate(data, y, is_fit, is_test)
    rows.append(score("Route delay rate", y_test, test_rate,
                      pick_threshold(y_val, val_rate, TARGET_DELAY_RECALL)))

    for name, features in MODELS.items():
        print(f"=== Baseline: logistic regression ({name} features) ===")
        lr = baselines.logistic_regression(data[features], y, is_fit)
        threshold = pick_threshold(y_val, lr.predict_proba(data.loc[is_val, features])[:, 1], TARGET_DELAY_RECALL)
        rows.append(score(f"Logistic regression ({name})", y_test,
                          lr.predict_proba(data.loc[is_test, features])[:, 1], threshold))
    return rows


def score(name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {"model": name,
            "ROC-AUC": round(roc_auc_score(y_true, probs), 4),
            "avg precision": round(average_precision_score(y_true, probs), 4),
            "accuracy": round((preds == y_true.to_numpy()).mean(), 3),
            "delayed caught": round(recall_score(y_true, preds, zero_division=0), 3),
            "on-time correct": round(recall_score(y_true, preds, pos_label=0, zero_division=0), 3)}


def train_model(X, y, is_fit, is_val):
    # Weight delayed flights up to balance the ~79/21 class split (like class_weight="balanced")
    scale_pos_weight = (y[is_fit] == 0).sum() / (y[is_fit] == 1).sum()
    model = XGBClassifier(
        n_estimators=3000,          # upper bound; early stopping picks the real number
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        enable_categorical=True,    # AIRLINE, ORIGIN, DEST are pandas categories
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1)
    model.fit(X[is_fit], y[is_fit], eval_set=[(X[is_val], y[is_val])], verbose=200)
    print("Best number of trees:", model.best_iteration + 1)

    # Pick the decision threshold on the validation set (never the test set),
    # as the highest cutoff that still catches TARGET_DELAY_RECALL of delayed flights
    val_probs = model.predict_proba(X[is_val])[:, 1]
    threshold = pick_threshold(y[is_val], val_probs, TARGET_DELAY_RECALL)
    print(f"Threshold for {TARGET_DELAY_RECALL:.0%} delayed recall on validation: {threshold:.3f}")
    return model, threshold


def pick_threshold(y_true, probs, target_recall):
    """Highest probability cutoff whose recall on delayed flights is >= target_recall."""
    _, recall, thresholds = precision_recall_curve(y_true, probs)
    # recall[i] is the recall at thresholds[i]; recall falls as the threshold rises
    meets_target = recall[:-1] >= target_recall
    return float(thresholds[meets_target].max())


def save_model(name, model, features, threshold, data):
    """Model file + the metadata the app needs to build matching inputs."""
    model.save_model(MODELS_DIR / f"{name}.json")
    categories = {c: data[c].cat.categories.tolist() for c in features if str(data[c].dtype) == "category"}
    meta = {"features": features, "threshold": threshold, "categories": categories,
            "target_delay_recall": TARGET_DELAY_RECALL,
            # Training up-weights delayed flights, which inflates predicted probabilities.
            # The app divides the predicted odds by this to get real probabilities.
            "scale_pos_weight": float(model.get_params()["scale_pos_weight"])}
    (MODELS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"Saved models/{name}.json")


if __name__ == "__main__":
    main()
