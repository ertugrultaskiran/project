#!/bin/bash
# Train all models sequentially

echo "=== Training All Models ==="
echo ""

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 1. Data preprocessing
echo "[1/5] Data preprocessing..."
jupyter nbconvert --to notebook --execute src/00_check_data.ipynb --output 00_check_data_executed.ipynb
echo "✓ Data preprocessing completed"
echo ""

# 2. Baseline model
echo "[2/5] Training baseline model..."
jupyter nbconvert --to notebook --execute src/01_baseline_tfidf_logreg.ipynb --output 01_baseline_executed.ipynb
echo "✓ Baseline model completed"
echo ""

# 3. LSTM model
echo "[3/5] Training LSTM model..."
jupyter nbconvert --to notebook --execute src/02_word2vec_lstm.ipynb --output 02_lstm_executed.ipynb
echo "✓ LSTM model completed"
echo ""

# 4. BERT model (optional)
echo "[4/5] Training BERT model (optional)..."
# jupyter nbconvert --to notebook --execute src/03_bert_transformer.ipynb --output 03_bert_executed.ipynb
echo "⊘ BERT model skipped (uncomment to run)"
echo ""

# 5. Evaluation
echo "[5/5] Running comprehensive evaluation..."
python src/07_model_evaluation.py
echo "✓ Evaluation completed"
echo ""

echo "=== All models trained successfully ==="
echo "Check the 'models/' and 'reports/' directories for outputs"

