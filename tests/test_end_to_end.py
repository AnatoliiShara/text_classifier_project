import pytest
import pandas as pd
from src.data.make_dataset import generate_synthetic_data
from src.models.train_model import load_data, preprocess_data, train_naive_bayes, train_random_forest, train_lightgbm, evaluate_model

def test_end_to_end():
    df = generate_synthetic_data(n_samples=100, n_categories=5)
    df.to_csv('data/raw/test_synthetic_data.csv', index=False)
    
    X, y = load_data('data/raw/test_synthetic_data.csv')
    (X_train, X_test, y_train, y_test) = preprocess_data(X, y)
    
    model = train_naive_bayes(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)
    
    assert isinstance(report, str)
    assert "accuracy" in report.lower()
    assert "f1-score" in report.lower()
    
# /home/anatolii-shara/Documents/mlops_experiments/text_classifier_project/src/text_classifier/__init__.py