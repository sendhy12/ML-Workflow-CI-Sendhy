import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_files_exist():
    """Test if required data files exist"""
    required_files = [
        './dataset_preprocessing/X_train.csv',
        './dataset_preprocessing/X_test.csv',
        './dataset_preprocessing/y_train.csv',
        './dataset_preprocessing/y_test.csv'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file {file_path} not found"

def test_data_loading():
    """Test if data can be loaded correctly"""
    try:
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv')
        
        # Basic checks
        assert not X_train.empty
        assert not X_test.empty
        assert not y_train.empty
        assert not y_test.empty
        
        # Shape consistency
        assert X_train.shape[1] == X_test.shape[1]
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")

def test_data_quality():
    """Test data quality"""
    X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
    y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')
    
    # Check for missing values
    assert not X_train.isnull().any().any(), "Training data contains null values"
    assert not y_train.isnull().any().any(), "Training labels contain null values"
    
    # Check label values
    unique_labels = y_train['target'].unique()
    assert set(unique_labels).issubset({0, 1}), "Labels should be 0 or 1"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])