from data_preprocessing import data_process
from train_models import train_models
from evaluate_models import predict_classifiers, evaluate_metrics
from visualization import dataplot_distribution, plot_confusion_matrix

#---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    print("--- Processing data...") # exec msg
    data = data_process(filepath="spam.csv") # cleans data, removes NaN values, removes punctuation, removes trailing whitespaces, splits data into classifiers and messages

    #print("--- Plotting true data...") # exec msg
    #dataplot_distribution(data, column='classifier', title='Spam and Non-Spam Distribution', chart='both') # charts true data, including count and proportion of true classifiers

    print("--- Training models...") # exec msg
    trained_models, (x_train, x_test, y_train, y_test), vectorizer, encoder = train_models(
        selected_models=["RandomForest", "NaiveBayes", "LogisticRegression"],
        data=data
    )

    print("--- Applying models...") # exec msg
    predictions = predict_classifiers(trained_models, x_test)

    print("--- Generating metrics...") # exec msg
    metrics = evaluate_metrics(y_test, predictions) # print scores (could make more readable later)
    

    """print("--- Generating confusion matrices...") # exec msg
    for model_name, y_pred in predictions.items(): # create a confusion matrix, this will demostrate FP, FN, TP, TNs
        plot_confusion_matrix(y_test, 
                              y_pred, 
                              title= f"{model_name} Confusion Matrix", 
                              labels=["Ham", "Spam"])"""
        
    print("Program finished.") # exec msg


    #--------------------------------------------Workflow Details----------------------------------
    #1. Process the data.
    #2. Create a bar and pie chart representing the quantity and ratio of each label.
    #3. Train the models based on the vectorised text (x) and actual labels (y), vectorization happens in "ml_preparation", and the dataset is also split into labels and vectorized text.
    #4. The trained model is then applied on vectorized email text.
    #5. Next, using the predicted labels and test labels, we compute the accuracy, precision, recall, and F1 score for each model.
    #6. Then, for each model, we plot a confusion matrix using the actual labels against the predicted labels, demonstrating false positives, false negatives, true positives, and true negatives.
    #7 Program complete.

    #--------------------------------------------Debugging----------------------------------

    #print(data.head)
    #print (predictions)
    #print(trained_models)
    #print(predictions)
    #print(metrics)