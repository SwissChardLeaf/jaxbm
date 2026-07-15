# jax-bm

JAX-based simulations of Boltzmann machines.

`jax-bm` aims to provide simple, composable, and `jit`-friendly building blocks
for simulating and training Boltzmann machines on top of JAX.

## Installation

From a clone of this repository:

```bash
pip install -e ".[dev]"
```

## Quick start

`sample_chain` is an array-in/array-out entry point: give it a symmetric
coupling matrix `W` and a bias vector `b`, and it infers everything else
(whether the machine is restricted, which conditional sampler to use, etc.)
from their shape and structure.

```python
import jax
import jax.numpy as jnp
from jax_bm import sample_chain

key = jax.random.PRNGKey(0)
n = 4
W = jnp.zeros((n, n))   # symmetric (n, n) coupling matrix, zero diagonal
b = jnp.zeros(n)        # bias vector, or None for no bias

x0 = jnp.ones(n)
final_x, samples = sample_chain(key, x0, W, b, steps=1, n_samples=100, mode="HIST")
```

See `sample_chain`'s docstring (`src/jax_bm/sample.py`) for the full set of
behaviors, including burn-in-only calls (`mode="LAST"`, the default, with
`n_samples=None`), running averages (`mode="MEAN"`), and mean correlation
matrices (`mode="CORR"`).

## Project layout

```
src/jax_bm/
  __init__.py
  sample.py      # sample_chain(): the public entry point, validation, and
                 # restricted (bipartite) detection
  _sampler.py    # conditional sampler builders (single-site Gibbs / block-Gibbs)
  _loop.py       # drivers: jax.lax.fori_loop (advance / running mean / running
                 # correlation) and jax.lax.scan (stacked trajectory)
archive/         # older helpers (statistics, plotting, misc utils), kept for
                 # reference but not part of the installable package
examples/        # worked examples (e.g. MNIST wake-sleep training), not part
                 # of the installable package
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
