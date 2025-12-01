"""
Custom Attention Layer Implementation
======================================

ORIGINAL CONTRIBUTION for Graduation Project

This module implements a custom attention mechanism for sequence classification.
Unlike standard approaches that use GlobalMaxPooling or last hidden state,
our attention layer learns to focus on important words/tokens dynamically.

Author: [Your Name]
Date: November 2025
"""

import tensorflow as tf
from tensorflow.keras import layers


class CustomAttentionLayer(layers.Layer):
    """
    Custom Attention Mechanism for LSTM/GRU outputs
    
    Mathematical Foundation:
    -----------------------
    Given LSTM outputs H = [h1, h2, ..., hT] where T is sequence length:
    
    1. Score calculation:
       score_t = tanh(W * h_t + b)
    
    2. Attention weights:
       α_t = exp(u^T * score_t) / Σ exp(u^T * score_i)
    
    3. Context vector (output):
       c = Σ α_t * h_t
    
    Parameters:
    -----------
    W : weight matrix (trainable)
    b : bias vector (trainable)
    u : context vector (trainable)
    
    Key Differences from Standard Approaches:
    -----------------------------------------
    - GlobalMaxPooling: Takes max across all timesteps (loses information)
    - GlobalAvgPooling: Simple average (no learned importance)
    - Our Attention: Learns which timesteps are important (interpretable)
    
    Benefits:
    ---------
    1. Better performance: Focuses on relevant parts
    2. Explainability: Attention weights show important words
    3. Flexibility: Works with variable-length sequences
    """
    
    def __init__(self, return_attention=False, **kwargs):
        """
        Initialize Custom Attention Layer
        
        Args:
            return_attention (bool): If True, also return attention weights
                                   (useful for visualization)
        """
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.return_attention = return_attention
    
    def build(self, input_shape):
        """
        Build layer parameters
        
        Args:
            input_shape: (batch_size, timesteps, features)
        """
        # input_shape: (None, timesteps, features)
        assert len(input_shape) == 3, "Input must be 3D: (batch, timesteps, features)"
        
        feature_dim = input_shape[-1]
        
        # Attention weight matrix
        self.W = self.add_weight(
            name='attention_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Attention bias
        self.b = self.add_weight(
            name='attention_bias',
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Context vector
        self.u = self.add_weight(
            name='attention_context',
            shape=(feature_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(CustomAttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        """
        Forward pass
        
        Args:
            inputs: LSTM/GRU outputs (batch_size, timesteps, features)
            mask: Optional mask for padded sequences
        
        Returns:
            context_vector: Weighted sum (batch_size, features)
            attention_weights: (Optional) Attention distribution
        """
        # inputs shape: (batch_size, timesteps, features)
        
        # Step 1: Calculate attention scores
        # score = tanh(W * h + b)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # score shape: (batch_size, timesteps, features)
        
        # Step 2: Calculate attention weights
        # attention = softmax(u^T * score)
        attention_logits = tf.tensordot(score, self.u, axes=1)
        # attention_logits shape: (batch_size, timesteps)
        
        # Apply mask if provided (for padded sequences)
        if mask is not None:
            # Convert mask to float and expand dimensions
            mask = tf.cast(mask, tf.float32)
            # Set masked positions to large negative value
            attention_logits = tf.where(
                tf.cast(mask, tf.bool),
                attention_logits,
                tf.ones_like(attention_logits) * -1e9
            )
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_logits, axis=1)
        # attention_weights shape: (batch_size, timesteps)
        
        # Step 3: Compute context vector
        # Reshape attention weights for broadcasting
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)
        # shape: (batch_size, timesteps, 1)
        
        # Apply attention weights to inputs
        weighted_input = inputs * attention_weights_expanded
        # shape: (batch_size, timesteps, features)
        
        # Sum across timesteps to get context vector
        context_vector = tf.reduce_sum(weighted_input, axis=1)
        # shape: (batch_size, features)
        
        if self.return_attention:
            return context_vector, attention_weights
        else:
            return context_vector
    
    def compute_output_shape(self, input_shape):
        """Compute output shape"""
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        else:
            return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        """Get layer configuration for serialization"""
        config = super(CustomAttentionLayer, self).get_config()
        config.update({'return_attention': self.return_attention})
        return config


def visualize_attention(text, attention_weights, tokenizer, top_k=10):
    """
    Visualize attention weights for a given text
    
    Args:
        text (str): Input text
        attention_weights (np.array): Attention weights (timesteps,)
        tokenizer: Keras tokenizer
        top_k (int): Number of top attended words to display
    
    Returns:
        List of (word, weight) tuples
    """
    # Tokenize text
    sequence = tokenizer.texts_to_sequences([text])[0]
    
    # Get words from indices
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    words = [reverse_word_index.get(idx, '<UNK>') for idx in sequence]
    
    # Get attention weights for actual words (not padding)
    word_attention = list(zip(words, attention_weights[:len(words)]))
    
    # Sort by attention weight
    word_attention_sorted = sorted(word_attention, key=lambda x: x[1], reverse=True)
    
    return word_attention_sorted[:top_k]


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("CUSTOM ATTENTION LAYER - ORIGINAL IMPLEMENTATION")
    print("=" * 70)
    print("\nThis is an ORIGINAL CONTRIBUTION to the graduation project.")
    print("\nKey Features:")
    print("  ✓ Custom attention mechanism (not pre-built)")
    print("  ✓ Mathematical foundation clearly documented")
    print("  ✓ Explainable AI capability")
    print("  ✓ Better than simple pooling methods")
    print("\nUsage in model:")
    print("""
    from custom_attention_layer import CustomAttentionLayer
    
    # Build model with attention
    inp = layers.Input(shape=(MAX_LEN,))
    x = layers.Embedding(...)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = CustomAttentionLayer()(x)  # ← OUR CUSTOM LAYER
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    """)
    print("=" * 70)

