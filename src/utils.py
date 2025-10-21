"""
Utility functions for text cleaning and preprocessing
"""
import re


def basic_clean(s: str) -> str:
    """
    Basic text cleaning function for English text.
    
    Args:
        s: Input string
        
    Returns:
        Cleaned string
    """
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-z0-9\-'\s]", " ", s)  # Keep letters, numbers, hyphens, apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

