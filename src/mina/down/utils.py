"""Utilities for converting MINA model outputs to AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def model_to_anndata(
    anndata_dict: dict[str, ad.AnnData],
    metadata: pd.DataFrame,
    model,
) -> ad.AnnData:
    """
    Combine a factor model and multiple pseudobulk views into a single AnnData object.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary of pseudobulk AnnData views. Keys are used to name
        entries in ``.obsm``.
    metadata : pandas.DataFrame
        Sample-level metadata indexed by sample ID.
    model : object
        Trained factor model exposing ``get_factors``, ``get_r2``,
        and ``get_weights`` methods.

    Returns
    -------
    anndata.AnnData
        AnnData object containing factor scores, metadata, gene loadings,
        and aligned pseudobulk matrices.
    """
    # ---------- 1) Collect unique sample IDs present in any view ----------
    if not anndata_dict:
        raise ValueError("anndata_dict is empty. Provide at least one AnnData view.")

    unique_samples_lists = [adata.obs_names.tolist() for adata in anndata_dict.values()]
    unique_samples = sorted(set().union(*unique_samples_lists))

    if not unique_samples:
        raise ValueError("No samples found across the provided AnnData views.")

    # ---------- 2) Align metadata to these samples ----------
    # Use reindex to avoid KeyError if some samples are missing in metadata; leave NaNs if absent.
    metadata_model = metadata.loc[unique_samples].copy()

    # ---------- 3) Factor scores (Z) ----------
    factors_dict = model.get_factors()
    if not isinstance(factors_dict, dict) or len(factors_dict) == 0:
        raise ValueError("model.get_factors() returned no groups.")

    # Clean factor names: remove spaces and ensure they start with 'Factor'
    def _clean_factor(name: object) -> str:
        s = str(name).replace(" ", "")
        return s if s.startswith("Factor") else f"Factor{s}"

    # Concatenate all groups along rows (samples)
    Z_list = []
    for _g, df in factors_dict.items():
        df_g = df.copy()
        df_g.columns = [_clean_factor(c) for c in df_g.columns]
        Z_list.append(df_g)

    Z_all = pd.concat(Z_list, axis=0)

    # Now restrict/reorder to the union of samples you already computed
    Z = Z_all.reindex(unique_samples)

    # ---------- 4) Explained variance per factor (R^2) ----------
    # model.get_r2() -> DataFrame with columns:
    # group, view, component, R2
    r2_df = model.get_r2()

    if not isinstance(r2_df, pd.DataFrame) or r2_df.empty:
        raise ValueError("model.get_r2() returned no R2 values.")

    required_cols = {"group", "view", "component", "R2"}
    missing = required_cols - set(r2_df.columns)
    if missing:
        raise ValueError(f"model.get_r2() is missing required columns: {missing}")

    r2_long = r2_df.copy()

    # Clean factor/component names so they match Z.columns
    r2_long["component"] = r2_long["component"].map(_clean_factor)

    # Encode R2 columns as "<view>:<group>", as in the previous implementation
    r2_long["view_group"] = r2_long["view"].astype(str) + ":" + r2_long["group"].astype(str)

    # Convert long format to factor × view_group matrix
    X = r2_long.pivot(
        index="component",
        columns="view_group",
        values="R2",
    )

    X.index.name = None
    X.columns.name = None

    # Ensure rows match the factor order in Z
    X = X.reindex(Z.columns)

    # ---------- 5) Feature weights (gene loadings) ----------
    # model.get_weights() -> dict: view_name -> DataFrame (features × factors)
    W_raw = model.get_weights()
    if not isinstance(W_raw, dict) or len(W_raw) == 0:
        raise ValueError("model.get_weights() returned no views.")

    # Transpose each to factors × features, then concat by keys -> MultiIndex (view_name, feature)
    W = {view: df.T for view, df in W_raw.items()}
    gene_weights = pd.concat(W.values(), axis=1)  # rows = factors, cols = features, stacked by view on the row index
    gene_weights.index = [
        _clean_factor(c) for c in gene_weights.index
    ]  # rows = factors, cols = features, stacked by view on the row index
    # Ensure rows of gene_weights (factors) match Z.columns order
    # gene_weights = gene_weights.T

    # ---------- 6) Build the AnnData (samples × factors) ----------
    # Convert Z to dense numpy. Handle potential sparse matrices defensively.
    Z_mat = Z.to_numpy()

    model_adata = ad.AnnData(
        X=Z_mat,
        obs=metadata_model,
        var=X,  # per-factor annotations; index are factor names
    )

    # ---------- 7) Store gene loadings in .varm (as array) and keep column names in .uns ----------
    model_adata.varm["gene_loadings"] = gene_weights.to_numpy()
    # If columns are a MultiIndex like (view, feature), keep only the feature level
    if isinstance(gene_weights.columns, pd.MultiIndex):
        gl_cols = gene_weights.columns.get_level_values(1).astype(str).tolist()
    else:
        gl_cols = [str(c) for c in gene_weights.columns]
    model_adata.uns["gene_loadings_columns"] = gl_cols

    # ---------- 8) Import all pseudobulk data into .obsm and copy feature names ----------
    # Also copy 'psbulk_cells' counts if present.
    factor_sample_index = model_adata.obs_names

    for key, temp_ad in anndata_dict.items():
        # Extract matrix (samples × features) as a dense DataFrame to align by sample
        X_view = temp_ad.X
        if sp.issparse(X_view):
            X_view = X_view.toarray()
        temp_df = pd.DataFrame(X_view, index=temp_ad.obs_names, columns=temp_ad.var_names)

        # Align rows to model_adata samples
        aligned_df = temp_df.reindex(factor_sample_index)

        # Save into .obsm and feature names into .uns
        model_adata.obsm[key] = aligned_df.to_numpy(copy=True)
        # Clean feature names from potential "view:feature" to just "feature"
        clean_cols = [(str(c).split(":", 1)[1] if ":" in str(c) else str(c)) for c in aligned_df.columns]
        model_adata.uns[f"{key}_columns"] = clean_cols

        # Copy contributing cell counts if available
        if "psbulk_cells" in temp_ad.obs.columns:
            model_adata.obs[f"{key}_n_cells"] = temp_ad.obs["psbulk_cells"].reindex(factor_sample_index).to_list()
        else:
            # Keep column for consistency but fill with NaN if missing
            model_adata.obs[f"{key}_n_cells"] = [np.nan] * model_adata.n_obs

    return model_adata


def restore_anns_factor(
    factor_names,
    annotation_source,
    *,
    strict=True,
):
    """
    Replace long guided factor names with their original annotation names.

    Assumes each annotation source maps to at most one factor name by suffix match.

    Parameters
    ----------
    factor_names : array-like
        Iterable of current factor names, e.g. model.factor_names.
    annotation_source : array-like
        Series/list/array with original annotation names, e.g. annotation.source.
    strict : bool
        If True, raise an error when one annotation source matches multiple factors.

    Returns
    -------
    restored : numpy.ndarray
        Factor names with matched long names replaced.
    """
    factor_names = np.asarray(factor_names, dtype=object)
    sources = pd.Series(annotation_source).dropna().drop_duplicates().astype(str)

    new_names = factor_names.copy()

    for source in sources:
        matches = [i for i, name in enumerate(factor_names) if str(name) == source or str(name).endswith(f"_{source}")]

        if len(matches) == 0:
            continue

        if strict and len(matches) > 1:
            raise ValueError(f"Annotation source {source!r} matched multiple factors: {factor_names[matches].tolist()}")

        for i in matches:
            new_names[i] = "Factor" + source

    return new_names


def split_by_view(arch_gex: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split a wide DataFrame into view-specific DataFrames.

    Column names are expected to follow the format ``"view:feature"``.

    Parameters
    ----------
    arch_gex : pandas.DataFrame
        DataFrame with columns encoded as ``view:feature``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping view names to DataFrames containing only
        features from that view.
    """
    # Check all column names contain ':'
    if not all(":" in col for col in arch_gex.columns):
        raise ValueError("All column names must be in the format 'view:feature'")

    # Split columns into (view, feature)
    split_cols = [col.split(":", 1) for col in arch_gex.columns]
    views, features = zip(*split_cols, strict=False)

    # Set MultiIndex
    arch_gex.columns = pd.MultiIndex.from_arrays([views, features], names=["view", "feature"])

    # Return dictionary of per-view DataFrames
    return {
        view: arch_gex.xs(view, axis=1, level="view") for view in arch_gex.columns.get_level_values("view").unique()
    }
