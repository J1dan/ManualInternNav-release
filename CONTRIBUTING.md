# Contributing

Thank you for helping improve ManualInternNav.

## Development setup

1. Fork and clone the repository.
2. Initialize every pinned submodule:

   ```bash
   git submodule update --init --recursive
   ```

3. Create a Python 3.10 environment and install the package without optional
   simulator dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e . pytest pyyaml
   ```

4. For ImagiNav training and evaluation, follow [ImagiNav/README.md](ImagiNav/README.md).
   Those workflows require a CUDA-capable system and additional model or
   dataset downloads; they are intentionally not part of the portable checks.

## Before opening a pull request

Run the portable checks locally:

```bash
python -m compileall -q internnav scripts tests
python -m pytest tests/unit_test -q
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Keep changes focused, update documentation when commands or configuration
paths change, and include tests for behavior changes. Do not commit API keys,
checkpoints, datasets, generated media, or machine-specific absolute paths.

## Hardware-dependent changes

For Isaac Sim, Habitat, GPU training, or real-robot changes, describe the
hardware and software versions used and include the exact validation command
and a concise result in the pull request. Maintainers review
hardware-specific evidence separately.

By submitting a contribution, you agree that it is licensed under this
repository's [MIT License](LICENSE).
