import pandas as pd
import string
import re


# -- process data
def data_process(filepath):
    """
    Load and clean dataset.

    Args:
    filepath (str): Path to CSV file to process.

    Returns:
    pd.DataFrame: Cleaned dataset with columns: 'classifier', 'message'.
    """

    data = pd.read_csv(filepath, encoding="latin-1") #open file
    data = data[['v1', 'v2']] #keep important columns
    data.columns = ['classifier', 'message'] #rename columns  
    data = data.dropna()   #remove missing values

    def clean_text(text):
        text = text.lower() # make text all lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        text = text.strip() # remove extra spaces
        return text
    
    data['message'] = data['message'].apply(clean_text) # apply cleaning

    return data