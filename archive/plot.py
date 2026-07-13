"""Matplotlib-based visualizations for Boltzmann-machine samples.

Matplotlib is imported lazily inside each plotting function so that the
rest of :mod:`jax_bm` can be used without pulling it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_samples(
    samples: ArrayLike,
    *,
    dims: int | Sequence[int] | None = None,
    max_dims: int = 8,
    figsize: tuple[float, float] | None = None,
    show_hist: bool = True,
    bins: int = 40,
    chain_alpha: float = 0.7,
    linewidth: float = 0.7,
    title: str | None = None,
) -> "Figure":
    """Plot per-component MCMC traces (and optional marginal histograms).

    Each selected vector component is shown as one row: a trace plot of all
    chains versus sample index on the left, and (by default) a horizontal
    marginal histogram on the right that shares its y-axis with the trace.
    Chains are color-coded so you can visually check whether they overlap.

    Parameters
    ----------
    samples
        Array-like of shape ``(n_chains, n_samples, vec_length)``.
    dims
        Which components to plot:

        - ``None`` (default): plot the first ``min(max_dims, vec_length)``.
        - ``int k``: plot the first ``k`` components.
        - sequence of ints: plot exactly those component indices.
    max_dims
        Cap on how many components are plotted by default. Ignored when
        ``dims`` is given explicitly.
    figsize
        Figure size in inches. Defaults to ``(10, 1.6 * n_rows)``.
    show_hist
        If ``True`` (default), draw a marginal histogram alongside each
        trace. Set to ``False`` for a denser trace-only view.
    bins
        Number of histogram bins.
    chain_alpha
        Opacity for trace lines and histogram bars (per chain).
    linewidth
        Line width for trace plots.
    title
        Optional suptitle for the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure. Call ``fig.savefig(...)`` or ``plt.show()``
        as desired; nothing is shown automatically.

    Raises
    ------
    ValueError
        If ``samples`` is not 3-dimensional, or if any requested dimension
        index is out of range.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(samples)
    if arr.ndim != 3:
        raise ValueError(
            "plot_samples expects samples of shape (n_chains, n_samples, vec_length); "
            f"got shape {arr.shape}."
        )
    n_chains, n_samples, vec_length = arr.shape

    if dims is None:
        dim_idx = list(range(min(max_dims, vec_length)))
    elif isinstance(dims, int):
        dim_idx = list(range(min(dims, vec_length)))
    else:
        dim_idx = list(dims)

    if len(dim_idx) == 0:
        raise ValueError("No dimensions selected to plot.")
    out_of_range = [d for d in dim_idx if not 0 <= d < vec_length]
    if out_of_range:
        raise ValueError(
            f"Dimension indices {out_of_range} are out of range for "
            f"vec_length={vec_length}."
        )

    n_rows = len(dim_idx)
    if figsize is None:
        figsize = (10.0, 1.6 * n_rows)

    if show_hist:
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=figsize,
            sharey="row",
            gridspec_kw={"width_ratios": [3, 1]},
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)

    cmap = plt.get_cmap("tab10")
    t = np.arange(n_samples)

    for row, d in enumerate(dim_idx):
        ax_trace = axes[row, 0]
        for c in range(n_chains):
            ax_trace.plot(
                t,
                arr[c, :, d],
                color=cmap(c % 10),
                alpha=chain_alpha,
                linewidth=linewidth,
                label=f"chain {c}" if row == 0 else None,
            )
        ax_trace.set_ylabel(f"x[{d}]")
        if row == n_rows - 1:
            ax_trace.set_xlabel("sample")
        else:
            ax_trace.tick_params(labelbottom=False)

        if show_hist:
            ax_hist = axes[row, 1]
            for c in range(n_chains):
                ax_hist.hist(
                    arr[c, :, d],
                    bins=bins,
                    color=cmap(c % 10),
                    alpha=0.5,
                    orientation="horizontal",
                    density=True,
                )
            ax_hist.tick_params(left=False, labelleft=False)
            if row == n_rows - 1:
                ax_hist.set_xlabel("density")
            else:
                ax_hist.tick_params(labelbottom=False)

    if n_chains > 1:
        fig.legend(
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            frameon=False,
            fontsize="small",
        )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
