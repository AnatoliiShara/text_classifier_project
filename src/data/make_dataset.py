import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_20newsgroups
import uuid

def generate_synthetic_data(n_samples=350, n_categories=7):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    selected_categories = np.random.choice(len(newsgroups.target_names), n_categories, replace=False)
    
    texts = []
    categories = []
    
    for _ in range(n_samples):
        category_idx = np.random.choice(selected_categories)
        category_texts = [text for text, target in zip(newsgroups.data, newsgroups.target) if target == category_idx]
        
        if category_texts:
            text = np.random.choice(category_texts)
            text = ' '.join(text.split()[:50])
            texts.append(text)
            categories.append(newsgroups.target_names[category_idx])
            
    df = pd.DataFrame({
        'id': [str(uuid.uuid4()) for _ in range(n_samples)],
        'text': texts,
        'category': categories
    })
    
    return df

data = generate_synthetic_data()
data.to_csv('data/raw/synthetic_data.csv', index=False)

print(f"Generated {len(data)} samples with {data['category'].nunique()} categories")
print(f"Data saved to data/raw/synthetic_data.csv")