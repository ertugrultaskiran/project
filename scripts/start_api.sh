#!/bin/bash
# Start the inference API

echo "Starting Ticket Classification API..."
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export FLASK_APP=src/06_inference_api.py
export FLASK_ENV=production

# Start API with gunicorn (production) or flask (development)
if command -v gunicorn &> /dev/null; then
    echo "Starting with Gunicorn (production mode)..."
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 src.06_inference_api:app
else
    echo "Starting with Flask (development mode)..."
    python src/06_inference_api.py
fi

