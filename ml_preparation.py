from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -- prepare data for machine learning
def prepareml(data, test_size=0.2):
    """
    Prepare dataset for machine learning.

    Args:
        data (pd.DataFrame): Cleaned dataset.
        test_size (float): Proportion of dataset used for testing.

    Returns:
        x_train, x_test, y_train, y_test, vectorizer, encoder
    """

    # ----- Convert text into TF-IDF features -----
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(data['message'])

    # ----- Encode labels -----
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['classifier'])

    # ----- Split dataset -----
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42
    )

    return x_train, x_test, y_train, y_test, vectorizer, encoder