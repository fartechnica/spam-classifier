#------------------------------------------Imports--------------------------------------------#
from data_preprocessing import data_process, plot_distribution
from ml_preparation import prepareml

#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":

# -------------------------1-------------------------------- #
    # process the data, enable this before running prepareml():
    #data = data_process() 

# -------------------------2-------------------------------- #
    # -OPTIONAL OPTIONAL OPTIONAL-
    # visualise the raw data 
    #plot_distribution(data, column='classifier', title='Spam and Non-Spam Distribution', chart='both') 

# -------------------------3-------------------------------- #
    # prepare the data for ML algorithm
    #, x_test, y_train, y_test, vectorizer, encoder = prepareml(data) 

#------------------------------------------Debugging--------------------------------------------#
    #print(data.head()) 
    #print("Training data shape:", x_train.shape) # verify training data shape output
    #print("Testing data shape:", x_test.shape) # verify testing data shape output