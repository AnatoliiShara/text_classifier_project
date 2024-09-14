import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.train_model import load_data, preprocess_data, train_naive_bayes, train_random_forest, train_lightgbm, evaluate_model

@pytest.fixture
def sample_data():
    X = [
        "This is a positive example",
        'This is a negative example',
        'Another positive example',
        'Yet another negative example'
    ]
    
    y = ['positive', 'negative', 'positive', 'negative']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    return X_vectorized, y

def test_train_naive_bayes(sample_data):
    X, y = sample_data
    model = train_naive_bayes(X, y)
    assert model.predict(X).shape == (4,)
    
def test_train_random_forest(sample_data):
    X, y = sample_data
    model = train_random_forest(X, y)
    assert model.predict(X).shape == (4,)
    
def test_train_lightgbm(sample_data):
    X, y = sample_data
    model = train_lightgbm(X, y)
    assert model.predict(X).shape == (4,)
    
def test_evaluate_model(sample_data):
    X, y = sample_data
    model = train_naive_bayes(X, y)
    report = evaluate_model(model, X, y)
    assert isinstance(report, str)
    assert 'accuracy' in report
    assert 'f1-score' in report
