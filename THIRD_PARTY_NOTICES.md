# Third-party notices

PairLabel's original application code is distributed under the MIT License in `LICENSE`.
Third-party source code and model weights retain their own licenses.

## Vendored SAM 2 source

- Upstream: https://github.com/facebookresearch/sam2
- Source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Location: `third_party/sam2/`
- Licenses: `third_party/sam2/LICENSE` (Apache License 2.0) and `third_party/sam2/LICENSE_cctorch`.
- The vendored `pyproject.toml` uses a reduced build-system declaration for the existing PyTorch environment. The installation script disables build isolation and automatic dependency resolution for this source package; use `scripts/install_sam.sh`.

## External runtimes and weights

CUDA llama.cpp is installed separately, not vendored. Model weights are not included in this repository.
See `MODEL_MANIFEST.json` for the tested runtime revisions, weight sources and available checksums.
Review the upstream model licenses before redistributing weights or using them in your own deployment.
