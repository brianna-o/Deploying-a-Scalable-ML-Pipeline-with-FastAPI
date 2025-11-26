import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed


def test_model():
    """
    # Testing that my  model is a RandomForestClassifier
    """
    X = np.array([[0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    # Verifying that the inference returns the same number of predictions as input rows
    """
    X = np.array([[0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == X.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_model_value():
    """
    # Checks that metric function returns numeric value for precision, recall, and fbeta
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
