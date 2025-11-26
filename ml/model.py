import pickle
from typing import Any, Dict, List, Tuple
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
import numpy as np
import pandas as pd


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Train a classification model and return it.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    RandomForestClassifier
        A fitted RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray,
    preds: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall and F-beta for a set of predictions.

    Parameters
    ----------
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run model inference and return predictions.

    Parameters
    ----------
    model : Any
        A trained model implementing `.predict`.
    X : np.ndarray
        Features to predict on.

    Returns
    -------
    np.ndarray
        Model predictions.
    """
    return model.predict(X)


def save_model(
    obj: Any,
    path: str,
) -> None:
    """
    Save a Python object (model, encoder, etc.) to disk with pickle.

    Parameters
    ----------
    obj : Any
        Object to save.
    path : str
        Path to the output file.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_model(path: str) -> Any:
    """
    Load a Python object (model, encoder, etc.) from disk.

    Parameters
    ----------
    path : str
        Path to the pickled object.

    Returns
    -------
    Any
        Loaded object.
    """
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def performance_on_categorical_slice(
    data,
    y_true,
    model,
    encoder,
    lb,
    feature,
    value,
    categorical_features,
    label="salary"
):
    """
    Compute precision, recall, and F1 for a single slice of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset (features + label column).
    y_true : np.ndarray
        True labels corresponding to `data`.
    model : Any
        Trained model.
    encoder : Any
        Fitted encoder for categorical features.
    lb : Any
        Fitted label binarizer.
    feature : str
        Name of the categorical feature to slice on.
    value : str
        Specific value of `feature` defining the slice.
    categorical_features : List[str]
        List of categorical feature names.
    label : str, optional
        Name of the label column, by default "salary".

    Returns
    -------
    precision: float
    recall: float
    fbeta: float
    """

    # filters rows in slice
    mask = data[feature] == value
    data_slice = data.loc[mask].copy()
    y_slice = y_true[mask]

    if data_slice.empty:
        return 0.0, 0.0, 0.0

    # Process the slice using existing encoder/lb
    X_slice, _, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference on slice
    preds_slice = inference(model, X_slice)

    # Compute metrics for slice
    precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
    return precision, recall, fbeta
