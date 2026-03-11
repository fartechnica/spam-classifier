from data_preprocessing import data_process
from ml_preparation import prepareml
from models.logistic_regression import train as train_lr
from models.naive_bayes import train as train_nb
from models.random_forest import train as train_rf

def train_models(selected_models=None, data=None, filepath="spam.csv"):
    """
    Train only the models specified in selected_models.

    Args:
        selected_models (list of str): Names of models to train.
            Options: 'LogisticRegression', 'NaiveBayes', 'RandomForest'.
            If None, train all models.
        filepath (str): CSV file path for preprocessing.

    Returns:
        dict: Trained models
        tuple: (x_train, x_test, y_train, y_test)
        vectorizer, encoder
    """
    # Default: train all
    all_models = {
        "LogisticRegression": train_lr,
        "NaiveBayes": train_nb,
        "RandomForest": train_rf
    }
    
    if selected_models is None:
        selected_models = list(all_models.keys())

    # ----- Preprocess and prepare data -----
    if data is None: # if data is not provided, process it here,
                     # but should ideally be done in main for modularity
        data = data_process(filepath)

    x_train, x_test, y_train, y_test, vectorizer, encoder = prepareml(data)

    # ----- Train only selected models -----
    trained_models = {}
    for model_name in selected_models:
        if model_name in all_models:
            trained_models[model_name] = all_models[model_name](x_train, y_train)
        else:
            print(f"Warning: '{model_name}' not recognized, skipping.")

    return trained_models, (x_train, x_test, y_train, y_test), vectorizer, encoder