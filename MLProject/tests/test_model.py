import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_creation():
    """Test if RandomForest model can be created"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    assert model is not None
    assert model.n_estimators == 10

def test_model_training():
    """Test if model can be trained on sample data"""
    # Create sample data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)

def test_feature_importance():
    """Test if feature importance is calculated"""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    importance = model.feature_importances_
    assert len(importance) == X.shape[1]
    assert np.isclose(np.sum(importance), 1.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])