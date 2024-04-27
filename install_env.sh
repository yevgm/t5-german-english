#!/bin/bash
python3 -m venv venv
source ./venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface
pip install transformers
pip install evaluate
pip install sentencepiece
pip install accelerate -U
pip install sacrebleu


echo Your environment has been successfully created.

