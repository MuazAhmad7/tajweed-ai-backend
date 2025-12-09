#!/bin/bash

# 🚀 Start Jupyter Notebook for Qalqalah Detection Pipeline
# This script starts Jupyter Lab in your browser

echo "🎯 Starting Tajweed AI Qalqalah Detection Pipeline"
echo "=================================================="

# Navigate to the backend directory
cd "$(dirname "$0")"

echo "📂 Current directory: $(pwd)"
echo "🔍 Looking for notebook: Qalqalah_Detection_Pipeline.ipynb"

if [ -f "Qalqalah_Detection_Pipeline.ipynb" ]; then
    echo "✅ Notebook found!"
else
    echo "❌ Notebook not found in current directory"
    exit 1
fi

echo ""
echo "🚀 Starting Jupyter Lab..."
echo "📱 Your browser will open automatically"
echo "🔗 If it doesn't, go to: http://localhost:8888"
echo ""
echo "💡 To stop the server: Press Ctrl+C"
echo "=================================================="

# Start Jupyter Lab (will open in browser automatically)
jupyter lab Qalqalah_Detection_Pipeline.ipynb --port=8888 --no-browser=false

echo ""
echo "👋 Jupyter Lab stopped. Goodbye!"
