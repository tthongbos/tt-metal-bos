# Current hierarchy

Repository subtree: `models/bos_model/oft`

Directory structure (folders, subfolders) and purpose:

- `__init__.py`: package initializer (exposes subpackages `reference` and `tt`).
- `reference/`:
  - `oft/`: reference PyTorch implementations and helpers used as ground-truth.
    - `model/`: PyTorch model implementations and utilities.
      - `torch/`: pure torch submodules (frontends, net definitions).
    - `data/`: dataset loaders and KITTI helpers.
      - `splits/`: text files listing KITTI indices for train/val/test/small/single.
    - `visualization/`: helper functions for plotting / visualizing outputs.
  - `__pycache__/`: compiled bytecode cache.

- `tt/`:
  - `ttnn_oft.py`: TTNN (Tenstorrent) accelerated versions of OFT layers.
  - `ttnn_oft_fixing.py`, `ttnn_oft_chay_duoc.py`: experimental/fix variants.
  - `__pycache__/`: compiled bytecode cache.

- `tests/`:
  - `test_configs/`: JSON files with small configs for unit/integration tests.
    - `default_infer_torch.json`
    - `oft_test_params.json`
  - `test_perspective.py`: unit tests for perspective/grid utilities.
  - `test_ttnn_oft_layer.py`: integration/test runner comparing torch vs ttnn OFT.
  - `infer_torch.py`, `infer_ttnn.py`: small inference utilities used for manual checks.

- `README.md`: this file


# TO DO
## all:
- create `tt/ttnn_oftnet.py`
- create `tests/test_ttnn_*.py` for each module
- create `run.py` convenience runner (optional)
- feel free to reuse the code from `reference` where appropriate

## Bach:
- optimize `ttnn_oft` (the class for the TT layers)

## Phu:
- optimize `ttnn_resnet`
