# Installation

## Dependencies Installation
The original environment used Python 3.9 and PyTorch 2.2.0 with CUDA 12.1.
The uv environment below uses Python 3.11 and CUDA 13.2 PyTorch wheels on Linux x86_64.

1.Clone our repository
git clone https://github.com/string-ellipses/DiffOSeg.git
```
cd DiffOSeg
```
2.Create and activate a virtual environment with [uv](https://docs.astral.sh/uv/):
```bash
uv sync --locked
source .venv/bin/activate
```

`pyproject.toml` defines the environment and explicitly selects the CUDA 13.2
index for PyTorch and torchvision. `uv.lock` pins the resolved dependency
versions and artifacts; `uv sync --locked` recreates this environment.

3.The original manual installation command remains available:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
uv pip install --python .venv/bin/python --torch-backend cu132 -r requirements-uv.txt
```

`requirements-uv.txt` is derived from the pip dependencies in `env.yml`.
It replaces the old PyTorch, CUDA 12 and Triton pins with the CUDA 13.2 stack,
omits unused torchaudio, and uses OpenCV 4.11.0.86 to retain NumPy 1.26.4.
The original `env.yml` and `requirements.txt` describe the historical environment.

For dependency changes, edit `pyproject.toml`, run `uv lock`, then `uv sync --locked`.
Keep the manual `requirements-uv.txt` pins aligned when changing its dependencies.

4.Check the environment:
```bash
uv pip check --python .venv/bin/python
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python ddpm_train.py --help
python ddpm_eval.py --help
```

## Dataset Download and Preperation
All the 2 datasets used in the paper can be downloaded from the following locations:

- LIDC-IDRI: [Google Drive](https://drive.google.com/file/d/1ilUq1j3_q0A-0QLHQfVoAei9GW6IOr3D/view?usp=drive_link)
- NPC-170: [Google Drive](https://drive.google.com/file/d/1dW1AN-g_6Bldnd3PC0vFT6sylyaE_NiJ/view?usp=drive_link)
