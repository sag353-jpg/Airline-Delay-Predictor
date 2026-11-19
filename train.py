# -- Build Pipeline --

from features import prepare_features

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from visuals import plot_results


def main():
    # Load and prepare dataset
    flights = prepare_features()

    # MONTH is useless because dataset is sequential
    features = ["AIRLINE", "STATE", "DISTANCE", "DAY_OF_WEEK", "MONTH", "SCHEDULED_DEPARTURE"]

    # Drop flights with no arrival delay recorded and define a delayed flight as delayed by >= 15 mins
    flights = flights.dropna(subset=["ARRIVAL_DELAY"]).copy()
    flights["delayed"] = (flights["ARRIVAL_DELAY"] >= 15).astype(int)

    # Count how many are delayed as a proportion
    print(flights["delayed"].value_counts(normalize=True))

    # Drop any row that has missing features
    data = flights.dropna(subset=features)

    # Train with 75/25 split
    X, y = data[features], data["delayed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, stratify=y, random_state=42)

    # The pipeline
    pipe = Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=20,
            random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    # Print model performance
    print(classification_report(y_test, pipe.predict(X_test)))

    # Display visualizations
    plot_results(pipe, X_test, y_test)


if __name__ == "__main__":
    main()