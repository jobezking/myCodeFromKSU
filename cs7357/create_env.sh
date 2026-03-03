#!/bin/bash

# ============================================================
#  Anaconda environment for Wav2Vec2 training (RTX 4060, CUDA 13)
# ============================================================

ENV_NAME="wav2vec2_env"

echo "Creating conda environment: $ENV_NAME"

# -----------------------------
# Create base environment
# -----------------------------
conda create -y -n $ENV_NAME python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# -----------------------------
# Install system-level deps
# -----------------------------
conda install -y -c conda-forge ffmpeg sox

# -----------------------------
# Install PyTorch (CUDA 12.1 build)
# -----------------------------
pip install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# HuggingFace + ML stack
# -----------------------------
pip install \
    transformers==4.40.0 \
    datasets \
    huggingface_hub \
    accelerate \
    soundfile \
    python-dotenv \
    packaging \
    tqdm \
    scikit-learn \
    wandb \
    ipython

# -----------------------------
# Optional: Apex (if needed)
# -----------------------------
# pip install git+https://github.com/NVIDIA/apex.git

echo "Environment '$ENV_NAME' created successfully."
echo "Activate it with:  conda activate $ENV_NAME"