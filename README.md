# jaxbm

JAX-based simulations of Boltzmann machines.

`jaxbm` aims to provide simple, composable, `jit`- and `vmap`-friendly building blocks
for simulating and training Boltzmann machines on top of JAX.

## Installation

From a clone of this repository:

```bash
pip install .
```

## Quick start

`jaxbm` exposes two functions: `BM_chain` for a
fully-connected Boltzmann machine, and `RBM_chain` for a restricted one
(separate visible/hidden units, block-Gibbs sampling). Both take a PRNG
`key`, an initial state, the model's `weights`/`bias` arrays, and `n_samples`
(required, no default -- `None` means "just advance the chain and return the
final state", otherwise a positive number of samples to draw). `steps`
(the number of chain-update steps between returned states) defaults to `1`.

```python
import jax
import jax.numpy as jnp
from jaxbm import BM_chain, RBM_chain

key = jax.random.PRNGKey(0)

# --- Fully-connected Boltzmann machine ---
n = 4
weights = jnp.zeros((n, n))   # symmetric (n, n) coupling matrix, zero diagonal
bias = jnp.zeros(n)           # bias vector, or None for no bias
x0 = jnp.ones(n)              # initial state, {-1, +1}-valued (spin=True, the default)

final_x, samples = BM_chain(key, x0, weights, bias, n_samples=100)
# final_x.shape == (4,), samples.shape == (100, 4)

# --- Restricted Boltzmann machine (visible/hidden layers) ---
n_v, n_h = 4, 3
weights_rbm = jnp.zeros((n_v, n_h))   # visible-hidden coupling matrix
bias_v, bias_h = jnp.zeros(n_v), jnp.zeros(n_h)
x_v0, x_h0 = jnp.ones(n_v), jnp.ones(n_h)

(final_v, final_h), (samples_v, samples_h) = RBM_chain(
    key, x_v0, x_h0, weights_rbm, bias_v, bias_h, n_samples=100
)
# (final_v, final_h) is the last state; samples_v.shape == (100, 4), samples_h.shape == (100, 3)
```

See `BM_chain` / `RBM_chain`'s docstrings (`jaxbm/sample.py`) for the full set
of behaviors, including validation rules, `spin`, and `clamp`.

## More guides

### Sampling modes
The `mode` selects how the
`n_samples` visited states are summarized. `mode` defaults to `"HIST"`, so
you can leave it out to just collect the trajectory:

```python
# mode defaults to "HIST": stack every sampled state -- shape (n_samples, n)
x, samples = BM_chain(key, x0, weights, bias, n_samples=100)

# mode="MEAN": running elementwise mean of the sampled states -- shape (n,)
x, x_mean = BM_chain(key, x0, weights, bias, n_samples=100, mode="MEAN")

# mode="CORR": running mean of outer(x, x) -- shape (n, n)
x, corr_mean = BM_chain(key, x0, weights, bias, n_samples=100, mode="CORR")

# n_samples=None (mode must also be None): advance 100 steps, no
# accumulation -- just the final state, returned bare (not a tuple)
x = BM_chain(key, x0, weights, bias, n_samples=None, steps=100)
```

None of the three modes above ever include the input state you passed in --
`samples` / `x_mean` / `corr_mean` only summarize the `n_samples` *new*
states produced by sampling.

`RBM_chain` supports the same modes, applied to the `(x_v, x_h)` pair
(`mode="CORR"` gives the mean of `outer(x_v, x_h)`, shape `(n_v, n_h)`).
Check the documentation for more details.

### `jit`

`n_samples`, `steps`, `mode`, `spin`, and `in_jit` all control Python-level
branching inside `BM_chain` / `RBM_chain`, so they must be passed as static
arguments under `jax.jit`. Eager input validation also isn't traceable, so
pass `in_jit=True` to skip it and make the whole call traceable.

```python
jitted_bm_chain = jax.jit(
    BM_chain, static_argnames=("n_samples", "steps", "mode", "spin", "in_jit")
)
final_x, samples = jitted_bm_chain(
    key, x0, weights, bias, n_samples=100, mode="HIST", in_jit=True
)
```

The same applies to `RBM_chain`.

### `vmap`

`BM_chain` / `RBM_chain` are `vmap`-able too. Run many independent
chains from the same model in parallel, batch over `key` (and optionally
`x0`, or even `weights`/`bias` for an ensemble of different models). A bare
`jax.vmap(...)` doesn't need `in_jit=True`:

```python
keys = jax.random.split(key, 8)  # 8 independent chains

run_many = jax.vmap(
    lambda k: BM_chain(k, x0, weights, bias, n_samples=100, mode="HIST")
)
final_xs, all_samples = run_many(keys)
# final_xs.shape == (8, 4), all_samples.shape == (8, 100, 4)
```

`in_jit=True` *is* needed as soon as `jit` enters the picture, though --
whether directly or wrapping a `vmap` (e.g. `jax.jit(jax.vmap(...))`):
```python
run_many_jit = jax.jit(jax.vmap(
    lambda k: BM_chain(k, x0, weights, bias, n_samples=100, mode="HIST", in_jit=True)
))
final_xs, all_samples = run_many_jit(keys)
# final_xs.shape == (8, 4), all_samples.shape == (8, 100, 4)
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT