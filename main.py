from data_preprocessing import data_process
from train_models import train_models
from evaluate_models import predict_classifiers, evaluate_metrics, predict_probabilities
from visualization import dataplot_distribution, plot_confusion_matrix, prCurve, rocCurve

#---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    print("--- Processing data...") # exec msg
    data = data_process(filepath="spam.csv") # cleans data, removes NaN values, removes punctuation, removes trailing whitespaces, splits data into classifiers and messages

    print("--- Training models...") # exec msg
    trained_models, (x_train, x_test, y_train, y_test), vectorizer, encoder = train_models( #trains the models, returns the trained models, the split data, and the vectorizer and encoder for later use
        selected_models=["RandomForest", "NaiveBayes", "LogisticRegression"],
        data=data
    )

    print("--- Predicting Probabilities...") # exec msg
    probabilities = predict_probabilities(trained_models, x_test) # generates predicted probabilities for the positive class (spam) for each model, this will be used for the ROC and PR curve.

    print("--- Predicting Classes...") # exec msg
    predictions = predict_classifiers(trained_models, x_test)

    # Visualisation. Enable if you want to see the dataplot, ROC and PR curve, and confusion matrix.

    print("--- Plotting true data...") # exec msg
    dataplot_distribution(data, column='classifier', title='Spam and Non-Spam Distribution', chart='both') # charts true data, including count and proportion of true classifiers

    print("--- Plotting ROC curve...") # exec msg
    for model_name, y_prob in probabilities.items(): # plot ROC curve for each model")
        rocCurve(y_test, y_prob, title=f"{model_name} ROC Curve")
        
    print("--- Plotting PR curve...") # exec msg
    for model_name, y_prob in probabilities.items(): # plot PR curve for each model")
        prCurve(y_test, y_prob, title=f"{model_name} Precision-Recall Curve")

    print("--- Generating Metrics...") # exec msg
    metrics = evaluate_metrics(y_test, predictions) # print scores for each model, including accuracy, precision, recall, and F1 score, this will be used to compare the models' performance.
    print(metrics)

    
    print("--- Generating confusion matrices...") # exec msg
    for model_name, y_pred in predictions.items(): # create a confusion matrix, this will demostrate FP, FN, TP, TNs
        plot_confusion_matrix(y_test, 
                              y_pred, 
                              title= f"{model_name} Confusion Matrix", 
                              labels=["Ham", "Spam"])
                              
        
    print("--- Program finished.") # exec msg

    #--------------------------------------------Workflow Details----------------------------------
    #1. Process the data.
    #2. Create a bar and pie chart representing the quantity and ratio of each label.
    #3. Train the models based on the vectorised text (x) and actual labels (y), vectorization happens in "ml_preparation", and the dataset is also split into labels and vectorized text.
    #4. Generate predicted probabilities for the positive class (spam) for each model, this will be used for the ROC curve and PR curve.
    #5. The trained model is then applied on vectorized email text.
    #6. Next, using the predicted labels and test labels, we compute the accuracy, precision, recall, and F1 score for each model.
    #7. Then, for each model, we plot a confusion matrix using the actual labels against the predicted labels, demonstrating false positives, false negatives, true positives, and true negatives.
    #8. We plot an ROC curve for each model using the predicted probabilities for the positive class (spam) and the true labels, this will show the trade-off between sensitivity and specificity for each model across different thresholds. This is not implemented in the current code but is planned for future development.
    #9 Program complete.

    #--------------------------------------------Debugging----------------------------------

    #print(data.head)
    #print (predictions)
    #print(trained_models)
    #print(predictions)
    #print(metrics)