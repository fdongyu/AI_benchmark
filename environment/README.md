# Recorded environment

- Platform: NERSC Perlmutter GPU nodes (NVIDIA A100)
- Python: 3.9.19, conda-forge build
- PyTorch: 2.6.0+cu124
- CUDA runtime packaged with PyTorch: 12.4
- NumPy: 1.24.3
- pandas: 2.2.2
- h5py: 3.11.0
- SciPy: 1.10.1
- matplotlib: 3.9.4
- transformers: 4.29.2

`pip-freeze.txt` is the exact package inventory observed at archive assembly.
`environment.yml` is the compact, portable Conda specification for the core
training, analysis, and test dependencies.
The NERSC environment path used by the batch scripts is
`/global/common/software/m3780/feng779/conda/rdycore-ai-gpu-py39`. For a public
release, build an equivalent CUDA 12.4/PyTorch 2.6 environment rather than
depending on that site-local path.
