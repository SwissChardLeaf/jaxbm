# jax-bm

JAX-based simulations of Boltzmann machines.

`jax-bm` aims to provide simple, composable, and `jit`-friendly building blocks
for simulating and training Boltzmann machines on top of JAX.

The only model is the fully-connected Boltzmann machine (symmetric couplings,
zero diagonal). Restricted Boltzmann machines are not a separate model class —
they are recovered as a fully-connected BM whose couplings have bipartite
block structure, together with the block-Gibbs sampler in `jax_bm.sampling`.

## Status

Early scaffolding. APIs are unstable.

## Installation

From a clone of this repository:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import jax
from jax_bm.models import BoltzmannMachine

key = jax.random.PRNGKey(0)
bm = BoltzmannMachine.init(key, n=64)
```

## Project layout

```
src/jax_bm/
  __init__.py
  models/        # fully-connected BoltzmannMachine
  sampling.py    # single-site Gibbs and bipartite block-Gibbs (RBM-style)
  training.py    # CD-k, PCD, and related training loops
  utils.py       # misc helpers (e.g. sigmoid, Bernoulli sampling)
tests/           # pytest-based unit tests
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
