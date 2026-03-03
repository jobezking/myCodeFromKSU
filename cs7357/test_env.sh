#!/bin/bash

conda activate wav2vec2_env
python - << 'EOF'
import torch, torchaudio
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF