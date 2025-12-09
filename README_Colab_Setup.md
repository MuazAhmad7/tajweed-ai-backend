# 🚀 Running Qalqalah Detection Pipeline in Google Colab

## Quick Start with Google Colab

### Option 1: Upload Notebook Directly
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "Upload" and select `Qalqalah_Detection_Pipeline.ipynb`
3. Run the first cell to install dependencies
4. Follow the notebook step by step

### Option 2: GitHub Integration
1. Upload your notebook to GitHub first
2. Go to Colab and select "GitHub" tab
3. Enter your repository URL
4. Select the notebook file

## 📁 Data Upload for Colab

Since Colab doesn't have access to your local files, you'll need to upload your data:

```python
# Add this cell at the beginning of your Colab notebook
from google.colab import files
import zipfile
import os

# Upload your audio files
print("📁 Upload your audio files (as a zip):")
uploaded = files.upload()

# Extract if it's a zip file
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('model_training/real_audio_extracted/')
        print(f"✅ Extracted {filename}")
    else:
        # Move individual files
        os.makedirs('model_training/real_audio_extracted/', exist_ok=True)
        os.rename(filename, f'model_training/real_audio_extracted/{filename}')

print("🎵 Audio files ready for analysis!")
```

## 🔧 Colab-Specific Modifications

Add this cell after the imports to handle Colab environment:

```python
# Colab-specific setup
import sys
if 'google.colab' in sys.modules:
    print("🌐 Running in Google Colab")
    
    # Install additional packages that might not be available
    !pip install montreal-forced-alignment
    
    # Mount Google Drive (optional, for persistent storage)
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Set up directories for Colab
    PROJECT_ROOT = Path('/content')
    MODEL_TRAINING_DIR = PROJECT_ROOT / "model_training"
    AUDIO_DIR = MODEL_TRAINING_DIR / "real_audio_extracted"
    OUTPUT_DIR = PROJECT_ROOT / "notebook_outputs"
    
    # Create directories
    MODEL_TRAINING_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
else:
    print("💻 Running locally")
```

## 🎯 Benefits of Colab
- ✅ Free GPU/TPU access
- ✅ Pre-installed ML libraries
- ✅ No local setup required
- ✅ Easy sharing and collaboration
- ✅ Automatic saving to Google Drive

## 📊 Limitations
- ⚠️ Session timeout after inactivity
- ⚠️ Need to re-upload data each session
- ⚠️ Limited persistent storage
