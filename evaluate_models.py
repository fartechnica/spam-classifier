from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----- Predict labels based on testing data-----
def predict_classifiers(models, x_test): # // apply the model to a set of test data (x)
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

# ----- Predict probabilities for positive class -----
def predict_probabilities(models, x_test): 
    """
    Generate predicted probabilities from trained models.

    Args:
        models (dict): Dictionary of trained models
        x_test: Feature matrix for testing
    Returns:
        dict: Mapping of model names to predicted probabilities
    """

    positiveProbabilities = {}
    for name, model in models.items():
        positiveProbabilities[name] = model.predict_proba(x_test)[:, 1] # Assuming binary classification, get probability of positive class]
    return positiveProbabilities

# ----- Evaluate model performance -----
def evaluate_metrics(y_test, predictions, decimal=3): 
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