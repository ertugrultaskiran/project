"""
Unit tests for models
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from src.utils import basic_clean

class TestDataPreprocessing:
    def test_basic_clean(self):
        """Test text cleaning function"""
        text = "Hello WORLD! Check http://example.com @user #hashtag"
        cleaned = basic_clean(text)
        
        assert cleaned.islower()
        assert "http" not in cleaned
        assert "@user" not in cleaned
        assert "#hashtag" not in cleaned
        
    def test_empty_string(self):
        """Test empty string handling"""
        assert basic_clean("") == ""
        assert basic_clean("   ") == ""

class TestModelPredictions:
    @pytest.fixture
    def sample_data(self):
        """Sample test data"""
        return pd.DataFrame({
            'text': [
                "I need to reset my password",
                "My laptop is not working",
                "Please approve my purchase request"
            ],
            'expected_label': [
                "Access",
                "Hardware",
                "Purchase"
            ]
        })
    
    def test_baseline_predictions(self, sample_data):
        """Test baseline model predictions"""
        import pickle
        
        with open("../models/baseline_tfidf_logreg.pkl", "rb") as f:
            model = pickle.load(f)
        
        predictions = model.predict(sample_data['text'])
        assert len(predictions) == len(sample_data)
        assert all(isinstance(p, str) for p in predictions)
    
    def test_lstm_predictions(self, sample_data):
        """Test LSTM model predictions"""
        from tensorflow.keras.models import load_model
        from tensorflow.keras.utils import pad_sequences
        import pickle
        
        # Load model and tokenizer
        model = load_model("../models/word2vec_lstm_model.h5")
        with open("../models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        # Preprocess
        sequences = tokenizer.texts_to_sequences(sample_data['text'])
        padded = pad_sequences(sequences, maxlen=80, padding="post")
        
        # Predict
        predictions = model.predict(padded)
        
        assert predictions.shape[0] == len(sample_data)
        assert predictions.shape[1] > 0  # Number of classes
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Softmax output

class TestAPI:
    def test_health_endpoint(self):
        """Test API health check"""
        # This would require running the API
        pass
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        # This would require running the API
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

