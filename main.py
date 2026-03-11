import pandas as pd
import matplotlib.pyplot as plt
import string

# data preprocessing

data = pd.read_csv("spam.csv", encoding="latin-1") #open file
data = data[['v1', 'v2']] #keep important columns
data.columns = ['classifier', 'message'] #rename columns  
data = data.dropna()   #remove missing values

# check class distribution, plot charts

def plot_distribution(data, column, title, colors=['#0352fcd1', '#fc2803c8'], chart='both'):
    
    """
    params:

    1. data - pandas DataFrame
    2. column - string
    3. title - string
    4. colors - list of colors for cats
    5. chart - 'bar', 'pie', 'both' (default)
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
        

# use below to bring up the distribution charts.

# plot_distribution(data, column='classifier', title='Spam and Non-Spam Distribution', chart='both')