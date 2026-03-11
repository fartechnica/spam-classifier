from sklearn.linear_model import LogisticRegression

def train(x_train, y_train):
    """
    Train a Logistic Regression model.
    
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model