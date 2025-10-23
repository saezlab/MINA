# Dependencies
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Plotting


def plot_view_samples(
    anndata_dict, min_samples, table=False, figsize=(5, 5), dpi=100, ax=None, return_fig=False, **kwargs
):
    """
    Quality Control plot to assess the quality of the obtained pseudobulk samples.

    Parameters
    ----------
    anndata_dict : AnnData
        Dictionary of AnnData objects.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    return_fig : bool
        Whether to return a Figure object or not.
    kwargs : dict
        Other keyword arguments are passed through to seaborn.scatterplot().

    Returns
    -------
    fig : Figure, None
        If return_fig, returns Figure object.
    """
    log_view_counts = []
    view_counts = []
    view_samples = []
    view_names = []
    for x, y in anndata_dict.items():
        df = y.var.copy()

        # Transform to log10
        log_view_counts.append(np.log10(sum(df["total_counts"])))
        view_counts.append(sum(df["total_counts"]))
        view_samples.append(y.shape[0])
        view_names.append(x)

    data = pd.DataFrame(
        list(zip(view_samples, view_counts, log_view_counts, strict=False)),
        columns=["Samples", "Total Counts", "Log Total Counts"],
        index=view_names,
    )

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    sns.scatterplot(x="Samples", y="Log Total Counts", hue=data.index, ax=ax, data=data, zorder=1, **kwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title="View")
    ax.axvline(x=min_samples, c="gray", ls="--")
    ax.set_xlabel("Total samples per view")
    ax.set_ylabel("Log10 total sum of counts")

    if return_fig:
        return fig

    if table:
        return data


def plot_view_genes(anndata_dict, min_genes, table=False, figsize=(5, 5), dpi=100, ax=None, return_fig=False, **kwargs):
    """
    Quality Control plot to assess the quality of the obtained pseudobulk samples.

    Parameters
    ----------
    anndata_dict : AnnData
        Dictionary of AnnData objects.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    return_fig : bool
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {``.pdf``, ``.png``, ``.svg``}.
    kwargs : dict
        Other keyword arguments are passed through to seaborn.scatterplot().

    Returns
    -------
    fig : Figure, None
        If return_fig, returns Figure object.
    """
    # Extract obs
    log_view_counts = []
    view_counts = []
    view_genes = []
    view_names = []
    for x, y in anndata_dict.items():
        df = y.var.copy()

        # Transform to log10
        log_view_counts.append(np.log10(sum(df["total_counts"])))
        view_counts.append(sum(df["total_counts"]))
        view_genes.append(y.shape[1])
        view_names.append(x)

    data = pd.DataFrame(
        list(zip(view_genes, view_counts, log_view_counts, strict=False)),
        columns=["Genes", "Total Counts", "Log Total Counts"],
        index=view_names,
    )

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    sns.scatterplot(x="Genes", y="Log Total Counts", hue=data.index, ax=ax, data=data, zorder=1, **kwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title="View")
    ax.axvline(x=min_genes, c="gray", ls="--")
    ax.set_xlabel("Total genes per view")
    ax.set_ylabel("Log10 total sum of counts")

    if return_fig:
        return fig

    if table:
        return data


def plot_sample_coverage(
    anndata_dict, threshold, proportion, table=False, figsize=(5, 5), dpi=100, return_fig=False, **kwargs
):
    """
    Visualize coverage for each AnnData in a dict; highlight & label samples under the given proportion threshold.

    Produces one figure per dict key (like your old loop).

    Parameters
    ----------
    anndata_dict : dict[str, AnnData]
        Dict with AnnData objects as values.
    threshold : float | dict[str, float]
        Gene expression threshold. If dict, must contain all keys of anndata_dict.
    proportion : float | dict[str, float]
        Minimum proportion of genes above `threshold`. If dict, must contain all keys.
    table : bool, default False
        If True, return a dict of DataFrames (per key) instead of plotting.
    figsize : tuple, default (5, 5)
        Figure size per subplot.
    dpi : int, default 100
        Figure DPI.
    return_fig : bool, default False
        If True, return a dict {key: Figure}.
    **kwargs :
        Passed to the underlying point plotting (matplotlib `ax.scatter`).

    Returns
    -------
    dict[str, pandas.DataFrame] if table=True
    dict[str, matplotlib.figure.Figure] if return_fig=True
    None otherwise
    """
    # Validate dict-style thresholds if provided
    if isinstance(threshold, dict):
        missing = set(anndata_dict.keys()) - set(threshold.keys())
        if missing:
            raise KeyError(f"'threshold' missing keys: {sorted(missing)}")
    if isinstance(proportion, dict):
        missing = set(anndata_dict.keys()) - set(proportion.keys())
        if missing:
            raise KeyError(f"'proportion' missing keys: {sorted(missing)}")

    tables = {}
    figs = {}

    for key, adata in anndata_dict.items():
        th = threshold[key] if isinstance(threshold, dict) else threshold
        prop = proportion[key] if isinstance(proportion, dict) else proportion

        counts = adata.X

        # Count genes > threshold per sample, robust to sparse/dense
        if hasattr(counts, "toarray") or str(type(counts)).endswith("spmatrix'>"):
            # Sparse path
            num_genes_above = np.asarray((counts > th).sum(axis=1)).ravel()
        else:
            # Dense path
            num_genes_above = np.sum(counts > th, axis=1)
            num_genes_above = np.asarray(num_genes_above).ravel()

        total_genes = counts.shape[1]
        prop_above = num_genes_above / float(total_genes)

        data = pd.DataFrame(
            {
                "Genes Above Threshold": num_genes_above,
                "Proportion Above Threshold": prop_above,
            },
            index=adata.obs.index,
        )
        tables[key] = data

        if table:
            continue

        # Create per-key figure/axis (like your old usage)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.grid(zorder=0)
        ax.set_axisbelow(True)

        below = data[data["Proportion Above Threshold"] < prop]
        above = data[data["Proportion Above Threshold"] >= prop]

        # Plot above-threshold samples in neutral gray (no legend)
        ax.scatter(
            above["Genes Above Threshold"],
            above["Proportion Above Threshold"],
            color="gray",
            s=40,
            zorder=1,
            label=None,
            **kwargs,
        )

        # Plot below-threshold samples individually (unique colors + legend)
        cmap = plt.get_cmap("tab10")
        for i, (sample, row) in enumerate(below.iterrows()):
            c = cmap(i % 10)
            ax.scatter(
                row["Genes Above Threshold"],
                row["Proportion Above Threshold"],
                color=c,
                s=70,
                zorder=2,
                label=sample,
                **kwargs,
            )

        # Threshold line
        ax.axhline(y=prop, c="gray", ls="--")

        # Labels & legend (only for below-threshold points)
        ax.set_xlabel("Genes Above Threshold")
        ax.set_ylabel("Proportion Above Threshold")
        if not below.empty:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                title="Below threshold",
            )

        # Title like your old suptitle usage
        fig.suptitle(f"{key}", fontsize=14, fontweight="bold")
        figs[key] = fig

    if table:
        return tables
    if return_fig:
        return figs


# Downstream plotting functions

# Associations

def plot_pval_tiles(p_df: pd.DataFrame, star_threshold: float = 0.05, ax=None, title: str | None = None):
    """
    Make a tile plot where tiles are colored by -log10(p) and tiles with p <= star_threshold
    are annotated with a star.

    Parameters
    ----------
    p_df : pd.DataFrame
        DataFrame of p-values (index = rows, columns = columns).
    star_threshold : float, default 0.05
        Threshold for plotting a star annotation.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this Axes; otherwise create a new figure and axes.
    title : str or None
        Optional title for the plot.
    """
    # Copy to avoid modifying the input
    p = p_df.copy()

    # Handle zeros or non-positive values to avoid -log10 issues
    # Replace any p <= 0 with the smallest positive float
    min_positive = np.nextafter(0, 1)
    p = p.mask(p <= 0, min_positive)

    # Compute -log10(p)
    neglog10 = -np.log10(p.astype(float))

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * neglog10.shape[1]), max(3.5, 0.6 * neglog10.shape[0])))
        created_fig = True

    # Plot tiles using imshow (matplotlib default colormap)
    im = ax.imshow(neglog10.values, aspect="auto")

    # Ticks & tick labels
    ax.set_xticks(range(neglog10.shape[1]))
    ax.set_yticks(range(neglog10.shape[0]))
    ax.set_xticklabels(neglog10.columns, rotation=45, ha="right")
    ax.set_yticklabels(neglog10.index)

    # Grid lines (optional, light)
    ax.set_xticks(np.arange(-0.5, neglog10.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, neglog10.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("-log10(p-value)")

    # Annotate stars for significant p-values
    star_mask = p.values <= star_threshold
    for i in range(star_mask.shape[0]):
        for j in range(star_mask.shape[1]):
            if star_mask[i, j]:
                ax.text(j, i, "★", ha="center", va="center")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if created_fig:
        return fig, ax
    return ax

# Functional enrichment


def plot_mcell_funcomics(
    result_dict: dict[str, dict[str, pd.DataFrame]],
    result_key: str = "pw_acts",
    pval_key: str = "pw_padj",
    p_threshold: float = 0.05,
    top_n: int = 10,
    cmap: str = "coolwarm",
    figsize: tuple = (14, 5),
    ytick_rotation: int = 0,
    use_var: bool = False,
):
    """
    Plot grouped heatmaps per view using the selected result (e.g. 'pw_acts'),

    filtering features based on adjusted p-value and showing shared color legend.

    Parameters
    ----------
    result_dict : dict
        Output of run_ulm_per_view — one entry per view.
    result_key : str
        Key inside each view result to plot ('pw_acts' or similar).
    pval_key : str
        Key inside each view result containing adjusted p-values.
    p_threshold : float
        Significance threshold for adjusted p-values.
    top_n : int
        Number of top significant features per view to show.
    cmap : str
        Colormap for heatmap.
    figsize : tuple
        Overall figure size.
    ytick_rotation : int
        Rotation angle for sample names on y-axis.
    use_var : bool
        If True, use variance instead of mean(abs) to rank features.
    """
    views = []
    filtered_data = {}

    # Step 1: collect views that pass p-value filtering
    for view, result in result_dict.items():
        data = result[result_key]  # samples × features
        pvals = result[pval_key]  # samples × features

        sig_mask = (pvals < p_threshold).any(axis=0)
        sig_features = sig_mask[sig_mask].index

        if len(sig_features) == 0:
            continue

        # Rank by mean(abs) or variance
        if use_var:
            feature_score = data[sig_features].var(axis=0)
        else:
            feature_score = data[sig_features].abs().mean(axis=0)

        top_features = feature_score.sort_values(ascending=False).head(top_n).index
        filtered_data[view] = data[top_features]
        views.append(view)

    n_views = len(views)
    if n_views == 0:
        print("No views with significant features found.")
        return

    # Compute global color scale
    all_vals = pd.concat(filtered_data.values(), axis=1)
    vmin, vmax = all_vals.min().min(), all_vals.max().max()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_views, wspace=0.4)
    heatmaps = []

    for i, view in enumerate(views):
        plot_data = filtered_data[view]

        ax = fig.add_subplot(gs[i])
        heatmap = sns.heatmap(
            plot_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,  # suppress individual colorbars
            ax=ax,
            xticklabels=True,
            yticklabels=(i == 0),
        )
        ax.set_title(view, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.tick_params(axis="y", labelsize=7, rotation=ytick_rotation)

        if i > 0:
            ax.set_ylabel("")
        heatmaps.append(heatmap)

    # Add shared colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # only needed for older versions of matplotlib
    fig.colorbar(sm, cax=cbar_ax, label=result_key)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for cbar
    plt.show()


# Multicellular information networks


def plot_mcell_network(
    df: pd.DataFrame,
    weight_col: str = "coef",
    abs_cutoff: float = 0.0,
    keep_negative: bool = True,
    edge_width_range: tuple = (0.8, 6),
    node_size: int = 1100,
    arrowsize: int = 18,
    reciprocal_curvature: float = 0.25,
    default_curvature: float = 0.04,
    positive_color: str = "tab:purple",
    negative_color: str = "tab:red",
    show_edge_labels: bool = False,
    label_fmt: str = "{:.2f}",
    title: str | None = None,
    save_path: str | None = None,
    edge_margin_factor: float = 0.55,
    arrows_on_top: bool = True,
):
    """PLACEHOLDER"""
    required_cols = {"target", "predictor", weight_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df[["target", "predictor", weight_col]].copy()
    d = d.dropna(subset=["target", "predictor", weight_col])
    d["weight"] = d[weight_col].astype(float)
    d = d[np.abs(d["weight"]) >= float(abs_cutoff)]
    if not keep_negative:
        d = d[d["weight"] >= 0]

    G = nx.DiGraph()
    for _, r in d.iterrows():
        G.add_edge(r["predictor"], r["target"], weight=float(r["weight"]))

    if G.number_of_edges() == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, "No edges after filtering.", ha="center", va="center")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.show()
        return G

    pos = nx.circular_layout(G)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_axis_off()
    node_coll = nx.draw_networkx_nodes(
        G, pos, node_color="#E9ECF6", node_size=node_size, edgecolors="#D2D6EA", linewidths=1.2, ax=ax
    )
    node_coll.set_zorder(2)
    label_dict = nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    for t in label_dict.values():
        t.set_zorder(3)

    edges = list(G.edges(data=True))
    weights = np.array([abs(a["weight"]) for _, _, a in edges], dtype=float)
    if weights.max() == weights.min():
        widths = np.full_like(weights, np.mean(edge_width_range), dtype=float)
    else:
        wmin, wmax = map(float, edge_width_range)
        widths = wmin + (weights - weights.min()) * (wmax - wmin) / (weights.max() - weights.min())

    colors = [positive_color if a["weight"] >= 0 else negative_color for _, _, a in edges]

    reciprocals = {tuple(sorted((u, v))) for u, v in G.edges() if G.has_edge(v, u)}
    curvatures = []
    for u, v, _ in edges:
        if tuple(sorted((u, v))) in reciprocals:
            curv = reciprocal_curvature if (u < v) else -reciprocal_curvature
        else:
            curv = default_curvature
        curvatures.append(curv)

    base = np.sqrt(node_size)
    margin = edge_margin_factor * base

    data = list(zip(edges, widths, colors, curvatures, strict=False))
    edge_z = 4 if arrows_on_top else 1

    def draw_batch(batch, rad):
        if not batch:
            return
        edgelist = [(u, v) for (u, v, _), _, _, _ in batch]
        widthlist = [w for _, w, _, _ in batch]
        colorlist = [c for _, _, c, _ in batch]
        arts = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=widthlist,
            edge_color=colorlist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrowsize,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=margin,
            min_target_margin=margin,
            ax=ax,
            alpha=0.95,
        )
        if arts is not None and arrows_on_top:
            try:
                for art in arts:
                    art.set_zorder(edge_z)
            except TypeError:
                arts.set_zorder(edge_z)

    neg = [(e, w, c, cv) for (e, w, c, cv) in data if cv < 0]
    posb = [(e, w, c, cv) for (e, w, c, cv) in data if cv > 0]
    flat = [(e, w, c, cv) for (e, w, c, cv) in data if abs(cv) <= 1e-9]
    draw_batch(neg, -abs(reciprocal_curvature))
    draw_batch(posb, abs(reciprocal_curvature))
    draw_batch(flat, default_curvature)

    if show_edge_labels:
        lbls = {(u, v): label_fmt.format(a["weight"]) for u, v, a in edges}
        edlbls = nx.draw_networkx_edge_labels(G, pos, edge_labels=lbls, font_size=8)
        for t in edlbls.values():
            t.set_zorder(5)

    if title:
        ax.set_title(title, pad=10)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.show()
    return G




