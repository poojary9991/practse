# test_model.py
# test_model.py
import pytest
from app import local_pipeline  # Import the local_pipeline from app.py

def test_local_sentiment_analysis():
    review = "I love this movie!"
    result = local_pipeline(review)
    assert result[0]['label'] in ['POSITIVE', 'NEGATIVE']
    assert 0.0 <= result[0]['score'] <= 1.0

def test_local_sentiment_analysis_new():
    """Test local sentiment analysis"""
    review = "I love this movie!"
    
    # Run the local sentiment analysis pipeline
    result = local_pipeline(review)
    
    # Assert that the result contains the expected keys
    assert 'label' in result[0], "Local result does not contain 'label'"
    assert 'score' in result[0], "Local result does not contain 'score'"
    
    # Check if the label is one of the expected sentiment labels
    assert result[0]['label'] in ['POSITIVE', 'NEGATIVE'], "Unexpected sentiment label"
    
    # Check if the score is a float
    assert isinstance(result[0]['score'], float), "Score should be a float"

    # Print result for manual inspection (optional)
    print("Local Analysis Result:", result)

