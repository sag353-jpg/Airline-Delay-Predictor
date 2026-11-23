from features import prepare_features, NULLABLE_FEATURES
from weather import WEATHER_FEATURES
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from visuals import plot_results
import numpy as np
import pandas as pd


def main():
    # Load and prepare dataset
    flights = prepare_features()

    features = ["AIRLINE", "ORIGIN", "DEST", "DISTANCE", "DAY_OF_WEEK", "MONTH",
                "DEP_MINUTES", "LEG_OF_DAY", "TURNAROUND_MIN",
                "DAY_OF_YEAR", "DAYS_TO_HOLIDAY", "SCHEDULE_PADDING",
                "ORIGIN_HOUR_DEPS", "ORIGIN_DAY_DEPS", "DEST_HOUR_ARRS",
                # Delay on the plane's previous flight that day. Only known once that flight
                # lands, so this model predicts on the day of travel, not at booking time.
                "PREV_ARRIVAL_DELAY", "PREV_DEPARTURE_DELAY", "PREV_DISRUPTED", "INBOUND_SLACK",
                # What's happening at the airports right now (flights 3h-1h before departure)
                "ORIGIN_RECENT_DEP_DELAY", "ORIGIN_RECENT_DEPS", "DEST_RECENT_ARR_DELAY"] + WEATHER_FEATURES

    # Drop flights with no arrival delay recorded
    flights = flights.dropna(subset=["ARRIVAL_DELAY"]).copy()

    # Define a delayed flight as delayed by >= 15 mins Count how many are delayed as a proportion
    flights["delayed"] = (flights["ARRIVAL_DELAY"] >= 15).astype(int)
    # Prints proportion of non-delayed to delayed. 81% not delayed, 19% not delayed
    # print(flights["delayed"].value_counts(normalize=True))

    # Drop any row that has missing non-weather features.
    # Weather gaps stay as NaN: XGBoost learns which way to send missing values.
    nullable = WEATHER_FEATURES + NULLABLE_FEATURES
    data = flights.dropna(subset=[f for f in features if f not in nullable]).copy()

    # Split by whole days, not individual flights. With a random flight-level split, flights
    # from the same storm day land in both train and test and the model "remembers" that day.
    # 25% of days -> test, 20% of the rest -> validation (early stopping + threshold).
    y = data["delayed"]
    days = np.random.default_rng(42).permutation(data["DAY_OF_YEAR"].unique())
    n_test, n_val = int(len(days) * 0.25), int(len(days) * 0.75 * 0.2)
    test_days, val_days = days[:n_test], days[n_test:n_test + n_val]
    is_test = data["DAY_OF_YEAR"].isin(test_days)
    is_val = data["DAY_OF_YEAR"].isin(val_days)
    is_fit = ~is_test & ~is_val

    X = data[features]
    X_fit, y_fit = X[is_fit], y[is_fit]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]

    # Weight delayed flights up to balance the ~81/19 class split (like class_weight="balanced")
    scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()

    # The pipeline
    pipe = Pipeline([
        ("model", XGBClassifier(
            n_estimators=2000,          # upper bound; early stopping picks the real number
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
            n_jobs=-1))
    ])
    pipe.fit(X_fit, y_fit, model__eval_set=[(X_val, y_val)], model__verbose=100)
    print("Best number of trees:", pipe.named_steps["model"].best_iteration + 1)

    # Pick the decision threshold on the validation set (never the test set),
    # as the highest cutoff that still catches TARGET_DELAY_RECALL of delayed flights
    val_probs = pipe.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_probs, TARGET_DELAY_RECALL)
    print(f"Threshold for {TARGET_DELAY_RECALL:.0%} delayed recall on validation: {threshold:.3f}")

    # Evaluate on the untouched test set with that threshold
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(classification_report(y_test, preds, target_names=["on time", "delayed"]))

    # Threshold-free metrics (best for comparing feature sets)
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Avg precision:", average_precision_score(y_test, probs))

    plot_results(y_test, preds, probs)

    # Look at performance of each feature
    importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))


# Share of delayed flights the model should catch. Higher = more delays caught,
# but more on-time flights wrongly flagged as delayed.
TARGET_DELAY_RECALL = 0.70


def pick_threshold(y_true, probs, target_recall):
    """Highest probability cutoff whose recall on delayed flights is >= target_recall."""
    _, recall, thresholds = precision_recall_curve(y_true, probs)
    # recall[i] is the recall at thresholds[i]; recall falls as the threshold rises
    meets_target = recall[:-1] >= target_recall
    return thresholds[meets_target].max()


if __name__ == "__main__":
    main()