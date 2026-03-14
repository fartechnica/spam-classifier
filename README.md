# Spam Classifier Project

This repository contains a simple Python workflow for training and evaluating machine learning models to classify SMS messages as **spam** or **ham** (not spam). The project demonstrates data preprocessing, feature extraction (TF-IDF), model training, evaluation, and basic visualization.

## What's Included?

- **Data preprocessing**: clean and normalize SMS text.
- **Feature extraction**: TF-IDF vectorization for machine learning.
- **Model training**: Logistic Regression, Naive Bayes, and Random Forest.
- **Evaluation**: accuracy, precision, recall, and F1 score.
- **Visualization**: confusion matrix and label distribution plots.

## Repository Structure

- `data_preprocessing.py` – loads and cleans the raw dataset.
- `ml_preparation.py` – vectorizes text and encodes labels, then splits train/test.
- `models/` – model training implementations:
  - `logistic_regression.py`
  - `naive_bayes.py`
  - `random_forest.py`
- `train_models.py` – orchestrates data prep and trains selected models.
- `evaluate_models.py` – generates predictions and computes evaluation metrics.
- `visualization.py` – plots label distributions and confusion matrices.
- `main.py` – example script showing a typical workflow.

**Run the main script**:

```bash
python main.py
```

This will:
- Load and clean `spam.csv`
- Train the configured models
- Evaluate their performance on a holdout test set

## Custom Usage

- Change the models trained by modifying `selected_models` in `main.py` or by calling `train_models(selected_models=[...])`.
- Add visualizations by uncommenting the plotting section in `main.py`.

## Notes

- The dataset is expected to be in `spam.csv` with columns `v1` (label) and `v2` (text).
- Models are trained on TF-IDF features and use scikit-learn estimators.
