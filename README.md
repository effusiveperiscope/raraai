# What is this?
A collection of tests and experiments for speech generation/conversion related ML tasks

# Installing dependencies (untested)
0. Use conda or a venv if you feel like it
1. Install `torch`, `torchaudio`, `torchvision` with a compatible CUDA version using the pytorch index (e.g. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`)
2. `pip install -r requirements.txt`

# Notable experiments
- `17_reexamining_rvq` is a SVC (singing voice conversion model) based loosely off of so-vits-svc 5.0 using the 48khz decoder from RVC. Instead of using whisper-medium features for input, it attempts to achieve speaker invariance by using a quantized representation of whisper-base features from a VQ-VAE trained in `stage1` (approach taken from Amphion's VeVo).
