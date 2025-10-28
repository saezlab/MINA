# Dependencies
import re

import pandas as pd
import scanpy as sc

# Upstream processing functions


# Generate study metadata from pseudobulks
def extract_metadata_from_obs(obs: pd.DataFrame, groupby: str, sort: bool = False) -> pd.DataFrame:
    """
    Extract group-level metadata by keeping only columns that have a single unique value per group.

    Parameters
    ----------
        obs (pd.DataFrame): The .obs DataFrame from an AnnData object.
        groupby (str): Column to group by (e.g., 'patient_id').
        sort (bool): Determine if natural sorting should be used

    Returns
    -------
        pd.DataFrame: Group-level metadata with consistent columns, naturally sorted by group ID.
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
    Splits an AnnData object by cell type, returning a dictionary of AnnData objects per cell type.

    Parameters
    ----------
    - pdata (AnnData): Input AnnData object with gene expression data.
    - grouping (str): Column in `pdata.obs` indicating cell type.

    Returns
    -------
    - celltype_adata_dict (dict): Dict with cell types as keys and AnnData objects as values.
    """
    if grouping not in pdata.obs.columns:
        raise ValueError(f"'{grouping}' not found in `pdata.obs`.")

    celltype_adata_dict = {}

    for cell_type in pdata.obs[grouping].unique():
        celltype_adata_dict[cell_type] = pdata[pdata.obs[grouping] == cell_type].copy()

    return celltype_adata_dict


def norm_log(anndata_objects, target_sum=1e6, exclude_highly_expressed=False, max_value=None, center=True):
    """
    Normalizes the total counts for each sample in each AnnData object, log-transforms, and scales (centers and scales) the data.

    Parameters
    ----------
    - anndata_objects (dict): Dictionary with cell types as keys and AnnData objects as values.
    - target_sum (float): The target total count per sample after normalization. Default is 1,000,000.
    - exclude_highly_expressed (bool): Whether to exclude highly expressed genes from normalization. Default is False.
    - max_value (float): Clip (truncate) to this maximum value after scaling to avoid outliers. Default is None (no clipping).

    Returns
    -------
    - None: The function modifies the input dictionary in place.
    """
    for _key, adata in anndata_objects.items():
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
