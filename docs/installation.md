# Installation

```bash
pip install tn-noise-learning            # or: uv add tn-noise-learning
pip install "tn-noise-learning[plot]"    # with matplotlib, for History.plot
```

Python 3.12 to 3.15. The package enables 64-bit JAX when it is imported; single precision underflows the
trace-preservation penalty.

## From source

```bash
git clone https://github.com/Algorithmiq/tn-noise-learning.git
cd tn-noise-learning
just build --all-groups   # needs uv and just; see the developer guide
```
