import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# -- Visualize the data --

def plot_results(y_test, preds, probs):
    # preds use the chosen threshold, so the confusion matrix matches the printed report
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, normalize="true", display_labels=["on time", "delayed"])
    RocCurveDisplay.from_predictions(y_test, probs)
    PrecisionRecallDisplay.from_predictions(y_test, probs)
    plt.show()
