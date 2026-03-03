#!/bin/bash

# 1. Create the Conda environment with Python 3.10
echo "Creating conda environment: speech_clf..."
conda create -n speech_clf python=3.10 -y

# 2. Initialize conda for the current shell session
source $(conda info --base)/etc/profile.d/conda.sh
conda activate speech_clf

# 3. Install PyTorch and Torchaudio [cite: 1, 8, 37]
# This setup assumes a CUDA 11.8 compatible GPU for training 
echo "Installing PyTorch and Torchaudio..."
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install Hugging Face Ecosystem [cite: 1, 41, 42]
echo "Installing Transformers and Datasets..."
pip install transformers datasets huggingface_hub

# 5. Install Data Science and Utility libraries 
echo "Installing Pandas, NumPy, Scikit-learn, and others..."
pip install pandas numpy scikit-learn tqdm packaging python-dotenv wandb ipython

echo "------------------------------------------------"
echo "Setup Complete! Activate your environment with:"
echo "conda activate speech_clf"
echo "------------------------------------------------"