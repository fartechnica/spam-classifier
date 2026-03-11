from sklearn.ensemble import RandomForestClassifier

def train(x_train, y_train):
    """
    Train a Random Forest model.
    
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model