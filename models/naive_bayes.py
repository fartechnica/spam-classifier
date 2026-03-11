from sklearn.naive_bayes import MultinomialNB

def train(x_train, y_train):
    """
    Train a Naive Bayes model.
    
    Returns:
        Trained MultinomialNB model
    """
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model