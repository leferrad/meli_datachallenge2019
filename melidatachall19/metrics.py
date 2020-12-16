"""Module to handle metrics for model evaluation"""

import numpy as np
from sklearn import metrics

# Metrics to be used to evaluate the models
# Based on a multi-class classification problem
METRICS = {
    "accuracy": lambda y_t, y_p: metrics.accuracy_score(y_t, y_p),
    "balanced_accuracy": lambda y_t, y_p: metrics.balanced_accuracy_score(y_t, y_p),
    "f1_micro": lambda y_t, y_p: metrics.f1_score(y_t, y_p, average="micro"),
}


def evaluate_metrics(x, y, model, vectorizer, label_encoder):
    """
    Function to evaluate a model (with a preprocessing done with vectorizer + label encoder),
    in terms of some pre-defined metrics that compare
    predicted classes from 'x' against a target 'y'.

    Parameters
    ----------
    x : {np.array}
        X array with text data (to be processed with vectorizer)
    y : {np.array}
        y array of target to predict
    model : {tf.keras.Model}
        model to be evaluated
    vectorizer : {tf.TextVectorization}
        vectorizer to convert text into integer tokens in x
    label_encoder : {sklearn.preprocessing.LabelEncoder}
        to convert labels in range [0, N-1]

    Returns
    -------
    dict, with entries per metric evaluated
    """
    # Transform text in 'x' into int tokens
    x = vectorizer(np.array([[r] for r in x])).numpy()
    # Convert labels in range [0, N-1]
    y = label_encoder.transform(y)
    # Get predictions (OHE)
    y_p = model.predict(x)
    # From OHE to best class
    y_p = np.argmax(y_p, axis=1)

    # Compute all metrics for y_true, y_pred
    results = dict()
    for k, func in METRICS.items():
        results[k] = func(y, y_p)

    # Also add loss for debugging purposes
    results["loss"] = model.evaluate(x, y)[0]

    return results
