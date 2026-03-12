import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -- create a bar chart or pie chart
def dataplot_distribution(data, column, title, colors=['#0352fcd1', '#fc2803c8'], chart='both'):
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


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", labels=None): 
    """
    Plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title (str): Chart title
        labels (list): Class labels (optional)
    """

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)

    plt.title(title)
    plt.show()