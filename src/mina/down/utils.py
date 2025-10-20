# Dependencies:
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# Utility Functions:
def model_to_anndata(
    anndata_dict: dict[str, ad.AnnData],
    metadata: pd.DataFrame,
    model,
) -> ad.AnnData:
    """
    Convert a factor model (e.g., MOFAFLEX) output + pseudobulk AnnData views into a single AnnData.

    Parameters
    ----------
    anndata_dict
        Dict of AnnData objects (each a pseudobulk view). Keys will be used to name .obsm entries.
        Each AnnData is expected to have:
          - obs index = sample IDs (pseudobulk rows)
          - var index = feature (gene) names
          - obs['psbulk_cells'] (optional): number of contributing single cells per sample (will be copied to .obs)
    metadata
        DataFrame with sample-level metadata, indexed by sample ID. Will be aligned to the union of samples.
    model
        Trained model exposing:
          - get_factors() -> Dict[group_name, pd.DataFrame]  # rows = samples, cols = factors
          - get_r2()      -> Dict[group_name, pd.DataFrame]  # rows = factors, cols = views/batches
          - get_weights() -> Dict[view_name, pd.DataFrame]   # rows = features, cols = factors

    Returns
    -------
    AnnData
        AnnData with:
          - X: samples × factors (factor scores)
          - obs: metadata aligned to samples in X
          - var: per-factor annotations (explained variance per view), with columns like "<view>:<group>"
          - varm['gene_loadings']: factors × (view:feature) weights (as a dense array)
          - uns['gene_loadings_columns']: list of (view:feature) column names for varm['gene_loadings']
          - obsm[<view>]: samples × features pseudobulk matrices aligned to samples in X
          - uns[<view>_columns]: feature (gene) names for each obsm matrix
          - obs[<view>_n_cells]: (if available) number of contributing cells per sample and view
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

    # Pick the first group by default (mirrors original 'group_1' behavior without hardcoding)
    group_name = list(factors_dict.keys())[0]
    Z = factors_dict[group_name].copy()

    # Keep only samples of interest and in the desired order
    # (Rows may be missing; reindex will introduce NaNs if so.)
    Z = Z.loc[unique_samples]

    # Clean factor names: remove spaces and ensure they start with 'Factor'
    def _clean_factor(name: object) -> str:
        s = str(name).replace(" ", "")
        return s if s.startswith("Factor") else f"Factor{s}"

    Z.columns = [_clean_factor(col) for col in Z.columns]

    # ---------- 4) Explained variance per factor (R^2) ----------
    r2_dict = model.get_r2()
    if group_name not in r2_dict:
        # Fall back to first available group in r2 if the same key is not present
        r2_group_name = list(r2_dict.keys())[0]
    else:
        r2_group_name = group_name

    X = r2_dict[r2_group_name].copy()
    # Ensure factor index names match Z's factor names using same cleaning
    X.index = pd.Index([_clean_factor(ix) for ix in X.index])

    # Append the (inferred) group name to column labels to keep provenance
    # (Mirrors "X.columns = X.columns + ':' + 'Chaffin'" from the prototype, but not hardcoded)
    X.columns = str(r2_group_name) + ":" + X.columns.astype(str)

    # Reorder rows of X to match the order of Z's columns (factors)
    # This ensures .var aligns with AnnData's columns (factors)
    # X = X.reindex(Z.columns)

    # ---------- 5) Feature weights (gene loadings) ----------
    # model.get_weights() -> dict: view_name -> DataFrame (features × factors)
    W_raw = model.get_weights()
    if not isinstance(W_raw, dict) or len(W_raw) == 0:
        raise ValueError("model.get_weights() returned no views.")

    # Transpose each to factors × features, then concat by keys -> MultiIndex (view_name, feature)
    W = {view: df.T for view, df in W_raw.items()}
    gene_weights = pd.concat(W)  # rows = factors, cols = features, stacked by view on the row index
    # Ensure rows of gene_weights (factors) match Z.columns order
    gene_weights = gene_weights.T

    # else: if no MultiIndex (unlikely for concat with keys), leave as-is

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


def split_by_view(arch_gex: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Splits a DataFrame whose columns are in the format 'view:feature' into a dictionary of DataFrames per view."""
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
