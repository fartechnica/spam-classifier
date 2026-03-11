import pandas as pd
import string
import matplotlib.pyplot as plt
import re



# -- process data
def data_process(filepath="spam.csv"):
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

# -- create plots
def plot_distribution(data, column, title, colors=['#0352fcd1', '#fc2803c8'], chart='both'):
    """
    Create plots.

    Args:
    data (pd.DataFrame): Dataset containing the column.
    column (str): Column name to plot.
    title (str): Chart title.
    colors (list): Colors for categories.
    chart (str): 'bar', 'pie', or 'both'
    """

    counts = data[column].value_counts() #count values

    if chart in ['bar', 'both']:

    # ------------- Bar Chart -------------
        ax = counts.plot(                              
            kind='bar',
            color=colors)

        for i, value in enumerate(counts):
            ax.text(i, value + 10, str(value), ha='center')
                
        plt.title(title + " (Counts)")
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.show()


    if chart in ['pie', 'both']:

    # ------------- Pie Chart -------------
        counts.plot(                              
            kind='pie',
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)
                
        plt.title(title + " (Percentages)")
        plt.ylabel('')
        plt.show()