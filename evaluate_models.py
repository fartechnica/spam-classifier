from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def predict_classifiers(models, x_test): # predict labels // apply the model to a set of test data (x)
    """
    Generate predictions from trained models.

    Args:
        models (dict): Dictionary of trained models
        x_test: Feature matrix for testing

    Returns:
        dict: Mapping of model names to predicted labels
    """
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(x_test)
    return predictions

def evaluate_metrics(y_test, predictions, decimal=3): # get metrics for each model
    """
    Compute accuracy, precision, recall, and F1 score for each model.

    Args:
        y_test: True labels
        predictions (dict): Predicted labels {model_name: y_pred}
        decimal (int): Number of decimal places to round the metrics
    Returns:
        dict: Metrics for each model {model_name: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}}
    """
    results = {}
    for name, y_pred in predictions.items():
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), decimal),
            "precision": round(precision_score(y_test, y_pred), decimal),
            "recall": round(recall_score(y_test, y_pred), decimal),
            "f1": round(f1_score(y_test, y_pred), decimal)
        }
    return results