#!/bin/bash

# 1. Create the Conda environment
echo "Creating conda environment: speech_clf..."
conda create -n speech_clf python=3.10 -y

# 2. Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate speech_clf

# 3. Install PyTorch + Torchaudio (Optimized for RTX 40-series/CUDA 12.4+)
# While your driver supports 13.0, 12.4 is the current stable target for 4060 performance.
echo "Installing PyTorch and Torchaudio..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Hugging Face & Data Science stack
echo "Installing project dependencies..."
pip install transformers datasets huggingface_hub \
            pandas numpy scikit-learn tqdm accelerate \
            packaging python-dotenv wandb ipython \
            git+https://github.com/NVIDIA/apex.git

echo "------------------------------------------------"
echo "Setup Complete!"
echo "Activate with: conda activate speech_clf"
echo "------------------------------------------------"