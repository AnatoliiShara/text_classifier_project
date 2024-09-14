import pandas as pd
import pytest
from data.make_dataset import generate_synthetic_data

def test_generate_synthetic_data():
    df = generate_synthetic_data(n_samples=100, n_categories=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert df['category'].nunique() == 5
    assert set(df.columns) == {'id', 'text', 'category'}
    assert df['id'].nunique() == 100
    