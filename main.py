from data_preprocessing import data_process, plot_distribution
from train_models import train_models

#---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # -------------------------1-------------------------------- 
    # Step 1: Process the data
    data = data_process() 

    # -------------------------2-------------------------------- 
    # Step 2: Visualise the raw data
    plot_distribution(data, column='classifier', title='Spam and Non-Spam Distribution', chart='both') 

    # -------------------------3-------------------------------- 
    # Step 3 Train models

    trained_models, (x_train, x_test, y_train, y_test), vectorizer, encoder = train_models(
        selected_models=["LogisticRegression", "RandomForest"],
        data=data
    )
    print("Models trained:", list(trained_models.keys()))

    #--------------------------------------------Debugging----------------------------------
    #print("Data preview:")
    #print(data.head()) 
    #print("Training data shape:", x_train.shape) # verify training data shape output
    #print("Testing data shape:", x_test.shape) # verify testing data shape output