"""
Simple Web App Launcher
========================

This script changes to the correct directory and launches the web app.
"""

import os
import sys

# Change to src directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
os.chdir(src_dir)

# Add src to path
sys.path.insert(0, src_dir)

# Import and run
print("=" * 70)
print("🚀 STARTING IT TICKET CLASSIFICATION WEB APP")
print("=" * 70)
print(f"\n📁 Working directory: {os.getcwd()}")
print(f"📁 Project root: {project_root}")

try:
    # Check if models exist
    models_dir = os.path.join(project_root, 'models')
    baseline_model = os.path.join(models_dir, 'baseline_tfidf_logreg.pkl')
    lstm_model = os.path.join(models_dir, 'word2vec_lstm_model.h5')
    
    print(f"\n🔍 Checking models...")
    print(f"   Baseline model: {'✓' if os.path.exists(baseline_model) else '✗'}")
    print(f"   LSTM model: {'✓' if os.path.exists(lstm_model) else '✗'}")
    
    if not os.path.exists(baseline_model) or not os.path.exists(lstm_model):
        print("\n⚠️  WARNING: Some models are missing!")
        print("   Please train models first:")
        print("   1. jupyter notebook src/01_baseline_tfidf_logreg.ipynb")
        print("   2. jupyter notebook src/02_word2vec_lstm.ipynb")
        print("\n   Continuing anyway (will show errors when classifying)...")
    
    print("\n" + "=" * 70)
    print("🌐 Starting Flask server...")
    print("=" * 70)
    print("\n📍 Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print("\n⏹  Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    # Import web app
    import web_app
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMissing packages? Try:")
    print("   pip install Flask Flask-Cors tensorflow scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

