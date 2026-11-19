import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# -- Visualize the data --

def plot_results(pipe, X_test, y_test):


    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, normalize="true")
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test)
    plt.show()
    classification_report(y_test, pipe.predict(X_test))



