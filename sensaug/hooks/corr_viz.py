"""Rendering for the augmentation cross-correlation matrix R.

Deliberately imports NOTHING from mmengine, mmseg or torch. Everything in
grad_sens_analysis.py is behind ``pytest.importorskip("mmengine")``, so a figure
that lived there could only be tested inside the cluster env -- a SLURM round-trip
to answer "does the heatmap look right". Keeping the render layer import-light
means tests/test_corr_viz.py runs on a laptop.

THE COLOR MAP. Correlation is a diverging quantity -- sign is meaning, zero is the
neutral -- so it gets a two-hue map with a NEUTRAL MIDPOINT, never a rainbow and
never a single hue. The midpoint is gray (#f0efec), not white: on a near-white
figure surface a white midpoint makes an r~0 cell indistinguishable from the paper
around it, which is the exact confusion the NaN masking below exists to remove.

The two arms are interpolated in OKLab rather than sRGB, and both poles are picked
at the same OKLab lightness (blue #0d366b L=0.338, red #6f0000 L=0.340). That is
what makes the arms symmetric: an r of -0.6 and an r of +0.6 render as equally
strong, so reading the matrix does not silently favour one sign. Interpolating the
same anchors in sRGB does not have that property.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# --- palette ----------------------------------------------------------------
# Chrome/ink, kept off the data colors: text never wears a series color.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
WARN_FILL = "#fab219"

# The diverging poles and the neutral midpoint.
POLE_NEG = "#0d366b"
MIDPOINT = "#f0efec"
POLE_POS = "#6f0000"
# Dropped ops (all-NaN rows) get the surface color plus a hatch -- readable as
# "no measurement", never as "measured, and it came out near zero".
DROPPED_FILL = "#e8e7e3"


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.abs(c) ** (1 / 2.4) - 0.055)


def _hex_to_oklab(value: str) -> np.ndarray:
    value = value.lstrip("#")
    rgb = np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    r, g, b = _srgb_to_linear(rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.array(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ]
    )


def _oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N, 3) OKLab -> (N, 3) sRGB in [0, 1]. Out-of-gamut results are clipped."""
    lab = np.atleast_2d(lab)
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l_**3, m_**3, s_**3
    rgb = np.stack(
        [
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        axis=-1,
    )
    return np.clip(_linear_to_srgb(rgb), 0.0, 1.0)


def _build_cmap(n_steps: int = 256) -> LinearSegmentedColormap:
    neg, mid, pos = (_hex_to_oklab(h) for h in (POLE_NEG, MIDPOINT, POLE_POS))
    half = n_steps // 2
    t = np.linspace(0.0, 1.0, half).reshape(-1, 1)
    lower = neg + (mid - neg) * t
    upper = mid + (pos - mid) * t
    colors = _oklab_to_rgb(np.vstack([lower, upper[1:]]))
    cmap = LinearSegmentedColormap.from_list("sensaug_corr", colors, N=n_steps)
    # A cell with no measurement must not land anywhere on the data ramp.
    cmap.set_bad(DROPPED_FILL)
    return cmap


CORR_CMAP = _build_cmap()


def dropped_ops(r: np.ndarray) -> np.ndarray:
    """Indices whose whole row is NaN -- the ops correlate() dropped for having no
    variance over the window. Derived from R itself so the figure cannot disagree
    with the matrix it is drawing."""
    return np.flatnonzero(~np.isfinite(np.asarray(r)).any(axis=1))


def render_matrix(
    r: np.ndarray,
    names,
    title: str,
    subtitle: str = "",
    mark: np.ndarray = None,
    warn: str = None,
    vlim: float = 1.0,
    cbar_label: str = "Pearson r",
) -> np.ndarray:
    """Render R as an (H, W, 3) uint8 array, ready for SummaryWriter.add_image.

    Args:
        r: (A, A) correlation matrix. NaN is a real value here (a dropped op), and
            is rendered as absent rather than as zero.
        names: length-A op names, in the canonical DIFF32_OPS
            order. Never reorder or cluster -- a moving axis would destroy the
            checkpoint-to-checkpoint comparability the whole figure exists for.
        mark: (A, A) bool. The only cells that get a printed number and a ring --
            normally the BH-FDR survivors. Annotating all A^2 cells (the previous
            behaviour) buries the handful you may actually act on among ~180 you
            may not. None marks nothing.
        warn: banner text stamped across the top. Used for the shared-factor alarm,
            so a matrix that must not be acted on says so on its own face and not
            only in the run log.
        vlim: colour limit, ALWAYS fixed by the caller and never derived from the
            data. Autoscaling per emission would make two checkpoints with
            different structure render identically.
    """
    r = np.asarray(r, dtype=float)
    n = len(names)
    dropped = set(dropped_ops(r).tolist())
    labels = [f"{nm} (dropped)" if i in dropped else nm for i, nm in enumerate(names)]

    fig, ax = plt.subplots(figsize=(max(7.0, n * 0.62), max(6.0, n * 0.56)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    masked = np.ma.masked_invalid(r)
    im = ax.imshow(
        masked, cmap=CORR_CMAP, vmin=-vlim, vmax=vlim, aspect="equal",
        interpolation="nearest",
    )

    # Hatch the unmeasured cells. set_bad already colors them; the hatch is the
    # redundant non-color channel, so the distinction survives a grayscale print
    # and a colorblind reader.
    for i in range(n):
        for j in range(n):
            if not np.isfinite(r[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor=INK_MUTED,
                        linewidth=0.0, alpha=0.55,
                    )
                )

    # Blank the diagonal. It is 1.0 by construction, so it carries no information
    # while rendering as the single most saturated band on the figure -- it draws
    # the eye away from the off-diagonal cells that are the entire point. Plain
    # fill, NOT the dropped-cell hatch, so "no information here" stays visually
    # distinct from "this op could not be measured".
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                facecolor=SURFACE, edgecolor=GRIDLINE, linewidth=0.4,
            )
        )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=INK_PRIMARY)
    ax.set_yticklabels(labels, fontsize=7, color=INK_PRIMARY)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDLINE)

    if mark is not None:
        mark = np.asarray(mark, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j or not mark[i, j] or not np.isfinite(r[i, j]):
                    continue
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor=INK_PRIMARY, linewidth=1.1,
                    )
                )
                # On-cell numbers sit on the diverging fill, so they switch with
                # the fill's lightness rather than wearing an ink token.
                ax.text(
                    j, i, f"{r[i, j]:.2f}",
                    ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="white" if abs(r[i, j]) > 0.55 * vlim else INK_PRIMARY,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8, color=INK_PRIMARY)
    cbar.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cbar.outline.set_edgecolor(GRIDLINE)

    ax.set_title(title, fontsize=10, color=INK_PRIMARY, pad=14 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=7, color=INK_MUTED, ha="left", va="bottom",
        )
    # tight_layout BEFORE the banner, with the top reserved when there is one:
    # placed after, the banner sits in figure coordinates and lands on top of the
    # title and subtitle instead of above them.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94) if warn else None)
    if warn:
        fig.text(
            0.5, 0.985, warn, ha="center", va="top", fontsize=8,
            color=INK_PRIMARY, fontweight="bold",
            bbox=dict(facecolor=WARN_FILL, edgecolor="none", pad=4.0),
        )

    return _figure_to_rgb(fig)


def _figure_to_rgb(fig) -> np.ndarray:
    """(H, W, 3) uint8 from a drawn figure.

    buffer_rgba, not tostring_rgb: the latter was deprecated in matplotlib 3.8 and
    REMOVED in 3.10, so it would take the whole correlation pipeline down on any
    env that has moved forward.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img


def markdown_report(
    names,
    r: np.ndarray,
    checkpoint: float,
    iteration: int,
    n_images: int,
    n_probes: int,
    dropped_names=(),
    magnitude_source: str = None,
    magnitude_mode: str = None,
    max_shared_loading: float = float("nan"),
    ci_lo: np.ndarray = None,
    ci_hi: np.ndarray = None,
    q: np.ndarray = None,
    survives: np.ndarray = None,
    top_n: int = 12,
) -> str:
    """A markdown table of the strongest pairs, for TensorBoard's TEXT tab.

    Puts the numbers that otherwise only exist in corr_bootstrap_log.txt next to
    the heatmap that summarises them, so reading R does not mean leaving the UI.
    All the significance columns are optional: with bootstrap=False they are simply
    absent rather than faked.
    """
    r = np.asarray(r, dtype=float)
    n = len(names)

    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if np.isfinite(r[a, b]):
                pairs.append((abs(r[a, b]), a, b))
    pairs.sort(reverse=True)

    header = ["pair", "r"]
    if ci_lo is not None and ci_hi is not None:
        header.append("95% CI")
    if q is not None:
        header.append("q (BH-FDR)")
    if survives is not None:
        header.append("survives")

    lines = [
        f"### R @ checkpoint {checkpoint:.0%} (iter {iteration})",
        "",
        f"- images **{n_images}** over **{n_probes}** probes",
        f"- magnitudes **{magnitude_source or 'unknown'} / {magnitude_mode or 'unknown'}**",
        f"- max shared-factor loading **{_num_cell(max_shared_loading, '{:.2f}')}**",
        f"- dropped: **{', '.join(dropped_names) if dropped_names else 'none'}**",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for _, a, b in pairs[:top_n]:
        row = [f"`{names[a]}` / `{names[b]}`", f"{r[a, b]:+.3f}"]
        if ci_lo is not None and ci_hi is not None:
            row.append(_ci_cell(ci_lo[a, b], ci_hi[a, b]))
        if q is not None:
            row.append(_num_cell(q[a, b], "{:.3g}"))
        if survives is not None:
            row.append("**yes**" if bool(survives[a, b]) else "no")
        lines.append("| " + " | ".join(row) + " |")

    if not pairs:
        lines.append("| _no finite cells_ | | |")

    return "\n".join(lines)


def _num_cell(value, fmt: str) -> str:
    value = float(value)
    return fmt.format(value) if np.isfinite(value) else "—"


def _ci_cell(lo, hi) -> str:
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"[{lo:+.2f}, {hi:+.2f}]"
