# Dependencies
import re

import pandas as pd
import scanpy as sc

# Upstream processing functions


# Generate study metadata from pseudobulks
def extract_metadata_from_obs(obs: pd.DataFrame, groupby: str, sort: bool = False) -> pd.DataFrame:
    """
    Extract group-level metadata from an observation table.

    Only columns with a single unique value per group are retained.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation metadata (e.g., ``AnnData.obs``).
    groupby : str
        Column used to define groups.
    sort : bool
        Whether to apply natural sorting to group identifiers.

    Returns
    -------
    pandas.DataFrame
        Group-level metadata table.
    """

    stable_cols = []

    for col in obs.columns:
        if col == groupby:
            continue

        # Group values and check if each group has only one unique value
        is_stable = (
            obs.groupby(groupby)[col]
            .apply(lambda x: x.dropna().nunique() <= 1)
            .all()
        )

        if is_stable:
            stable_cols.append(col)

    if not stable_cols:
        print("⚠️ No stable columns found other than the group ID.")

    # Now collect the first value from each group for these columns
    metadata = obs.groupby(groupby)[stable_cols].first().reset_index()

    metadata = metadata.set_index(groupby, drop=False)

    metadata.index.name = None

    # Sort naturally by extracting numeric part after underscore (if format like 'patient_11')
    if sort:

        def extract_number(x):
            match = re.search(r"(\d+)", x)
            return int(match.group(1)) if match else float("inf")

        metadata = metadata.sort_values(by=groupby, key=lambda col: col.map(extract_number)).reset_index(drop=True)

    return metadata


# From pseudobulk to list of anndatas


def split_anndata_by_celltype(pdata, grouping="cell_type"):
    """
    Split an AnnData object into multiple AnnData objects by cell type.

    Parameters
    ----------
    pdata : anndata.AnnData
        Input AnnData object.
    grouping : str
        Column in ``pdata.obs`` defining cell types.

    Returns
    -------
    dict[str, anndata.AnnData]
        Dictionary mapping cell types to AnnData objects.
    """

    if grouping not in pdata.obs.columns:
        raise ValueError(f"'{grouping}' not found in `pdata.obs`.")

    celltype_adata_dict = {}

    for cell_type in pdata.obs[grouping].unique():
        celltype_adata_dict[cell_type] = pdata[pdata.obs[grouping] == cell_type].copy()

    return celltype_adata_dict


def norm_log(anndata_dict, target_sum=1e6, exclude_highly_expressed=False, max_value=None, center=True):
    """
    Normalize, log-transform, and scale AnnData objects in place.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary of AnnData objects.
    target_sum : float
        Target total count per sample after normalization.
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes during normalization.
    max_value : float or None
        Maximum value after scaling to clip outliers.
    center : bool
        Whether to center features during scaling.

    Returns
    -------
    None
        The input dictionary is modified in place.
    """

    for _key, adata in anndata_dict.items():
        # Step 1: Perform total count normalization
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

        # Step 2: Log-transform the normalized data
        sc.pp.log1p(adata)

        # Step 3: Center and scale the data
        if center:
            sc.pp.scale(adata, max_value=max_value)  # Scaling and centering the data, optionally clipping large values

    # Optionally: Print confirmation that normalization, log-transformation, and scaling are complete
    if center:
        print(
            f"Normalization, log-transformation, and scaling complete for all AnnData objects with target_sum = {target_sum}."
        )

    else:
        print(f"Normalization and log-transformation complete for all AnnData objects with target_sum = {target_sum}.")
