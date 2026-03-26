import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, roc_auc_score

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

def rocCurve(y_true, y_prob, title="ROC Curve"):
        """
        Plot an ROC curve.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            title (str): Chart title
        """

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title(title, fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def prCurve (y_true, y_prob, title="Precision-Recall Curve"):
        """
        Plot a Precision-Recall curve.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            title (str): Chart title
        """

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2.5, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=11)
        plt.ylabel('Precision', fontsize=11)
        plt.title(title, fontsize=12)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()