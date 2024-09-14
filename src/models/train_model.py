import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Remove rows with NaN values
    df = df.dropna(subset=['text', 'category'])
    # Convert text to string type
    df['text'] = df['text'].astype(str)
    return df['text'], df['category']

def preprocess_data(X, y):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    return train_test_split(X_vectorized, y, test_size=0.2, random_state=42), vectorizer

def train_naive_bayes(X_train, y_train):
    nb = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
    grid_search = GridSearchCV(nb, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_lightgbm(X_train, y_train):
    lgbm = LGBMClassifier(random_state=42)
    param_grid = {
        'num_leaves': [31, 63, 127],
        'max_depth': [5, 10, -1],
        'learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(lgbm, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def main():
    # Load and preprocess data
    X, y = load_data('data/raw/synthetic_data.csv')
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(X, y)

    # Train models
    nb_model = train_naive_bayes(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)

    # Evaluate models
    models = {
        'Naive Bayes': nb_model,
        'Random Forest': rf_model,
        'LightGBM': lgbm_model
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nEvaluation for {name}:")
        report = evaluate_model(model, X_test, y_test)
        print(report)
        
        # Extract the weighted avg f1-score as the comparison metric
        f1_score = float(report.split('\n')[-2].split()[-2])
        if f1_score > best_score:
            best_score = f1_score
            best_model = (name, model)

    print(f"\nBest model: {best_model[0]} with F1-score: {best_score}")

    # Save the best model and vectorizer
    joblib.dump(best_model[1], 'models/best_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')

if __name__ == "__main__":
    main()