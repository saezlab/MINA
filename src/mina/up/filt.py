# Dependencies
import decoupler as dc
import numpy as np
import scanpy as sc

# Filtering views by number of cells in psbulk_cells column of .obs


def filter_anndata_by_ncells(anndata_dict, min_cells):
    """
    Filters out samples (rows) from AnnData objects in the dictionary where the number of cells (``psbulk_cells``) in ``.obs`` 
    is less than the specified threshold.

    Updates the .var attribute with total counts per gene.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with AnnData objects as values.
    min_cells : int or dict
        * If int, the same minimum number of cells is applied to all AnnData objects.
        * If dict, must have the same keys as anndata_dict, where each value is the minimum
          number of cells for that dataset.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    for key, adata in list(anndata_dict.items()):
        # Determine the threshold for this dataset
        if isinstance(min_cells, dict):
            if key not in min_cells:
                raise KeyError(f"Key '{key}' not found in min_cells dictionary.")
            min_cells_threshold = min_cells[key]
        else:
            min_cells_threshold = min_cells

        # Check if 'psbulk_cells' column exists in adata.obs
        if "psbulk_cells" in adata.obs.columns:
            # Identify samples with sufficient cells
            samples_to_keep = adata.obs[adata.obs["psbulk_cells"] >= min_cells_threshold].index

            # Filter the AnnData object
            adata_filtered = adata[samples_to_keep].copy()

            # Update the dictionary with the filtered AnnData object
            anndata_dict[key] = adata_filtered

            # Check if matrix is sparse or dense and compute gene counts
            if hasattr(adata_filtered.X, "toarray"):  # sparse
                gene_counts = np.array(adata_filtered.X.sum(axis=0)).flatten()
            else:  # dense
                gene_counts = adata_filtered.X.sum(axis=0)

            # Update .var with total counts
            adata_filtered.var["total_counts"] = gene_counts

        else:
            print(f"'psbulk_cells' column not found in AnnData.obs for key: {key}")


# Filtering by number of samples


def filter_views_by_samples(anndata_dict, min_rows):
    """
    Filters out AnnData objects in the dictionary that have fewer samples (rows) than the specified threshold.

    Also updates the .var attribute to include total counts per gene.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with AnnData objects as values.
    min_rows : int
        Minimum number of rows required for an AnnData object to remain in the dictionary.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    # Get keys of AnnData objects that don't meet the row count threshold
    keys_to_remove = [key for key, adata in anndata_dict.items() if adata.n_obs < min_rows]

    # Remove those AnnData objects from the dictionary
    for key in keys_to_remove:
        del anndata_dict[key]


# Filtering genes within a view per expression


def filter_genes_byexpr(anndata_dict, min_count, min_prop):
    """
    Filters genes in AnnData objects in the given dictionary based on count proportions, keeping all rows and filtering columns.

    Also updates the .var attribute with total counts per gene.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with cell types as keys and AnnData objects as values.
    min_count : int
        Minimum count threshold for filtering genes.
    min_prop : float
        Minimum proportion of samples (rows) where the count is >= min_count.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    for cell_type, adata in anndata_dict.items():
        min_count = np.clip(min_count, 0, None)
        # Extract the count matrix (adata.X)
        counts = adata.X

        # Calculate the proportion of rows (samples) where count >= min_count for each gene (column)
        total_rows = counts.shape[0]
        prop_values = np.sum(counts >= min_count, axis=0) / total_rows

        # Select genes (columns) where the proportion of samples >= min_count is >= min_prop
        selected_genes = prop_values >= min_prop

        # Compute lib_size if needed
        lib_size = np.sum(counts, axis=1)

        # CPM cutoff
        cpm_cutoff = dc.pp.anndata._cpm_cutoff(lib_size, min_count)

        # CPM mask
        cpm = dc.pp.anndata._cpm(counts, lib_size)
        sample_size = np.sum(cpm >= cpm_cutoff, axis=0)
        keep_cpm = sample_size >= total_rows * min_prop

        # Merge msks
        msk = keep_cpm & selected_genes

        # Update the AnnData object to keep only the selected genes (columns)
        adata_filtered = adata[:, msk].copy()  # Keep all rows, filter columns

        # Calculate total counts per gene for the filtered genes
        gene_counts = np.sum(adata_filtered.X, axis=0)

        # Update the .var DataFrame with total counts
        adata_filtered.var["total_counts"] = gene_counts

        # Update the dictionary with the filtered AnnData object
        anndata_dict[cell_type] = adata_filtered


# Filtering views by number of genes

def filter_views_by_genes(anndata_dict, min_genes_per_view):
    """
    Drops AnnData objects from the dictionary that have fewer than the specified number
    of genes (columns) after filtering.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with cell types as keys and AnnData objects as values.
    min_genes_per_view : int
        Minimum number of genes (columns) that must remain in an AnnData object for it to be kept.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    # Create a list to store keys of views that should be removed
    keys_to_remove = [key for key, adata in anndata_dict.items() if adata.n_vars < min_genes_per_view]

    # Remove the views (AnnData objects) with insufficient genes
    for key in keys_to_remove:
        del anndata_dict[key]


# Filtering samples within a view by coverage (number of genes covered)


def filter_samples_by_coverage(anndata_dict, threshold, min_prop):
    """
    Filters out samples in AnnData objects that do not have a sufficient proportion of genes with values greater than a specified threshold.

    Updates the dictionary in place and updates the .var attribute with total counts per gene.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with cell types as keys and AnnData objects as values.
    threshold : float
        The count threshold a gene value must exceed to be considered. Normally left at 0.
    min_prop : float
        Minimum proportion of genes that must exceed the threshold for a sample to be kept.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    # Validate dict-style thresholds if provided
    if isinstance(threshold, dict):
        missing = set(anndata_dict.keys()) - set(threshold.keys())
        if missing:
            raise KeyError(f"'threshold' missing keys: {sorted(missing)}")
    if isinstance(min_prop, dict):
        missing = set(anndata_dict.keys()) - set(min_prop.keys())
        if missing:
            raise KeyError(f"'proportion' missing keys: {sorted(missing)}")

    for cell_type, adata in anndata_dict.items():
        th = threshold[cell_type] if isinstance(threshold, dict) else threshold
        prop = min_prop[cell_type] if isinstance(min_prop, dict) else min_prop
        # Extract the count matrix (adata.X)
        counts = adata.X

        # Calculate the number of genes for each sample (row) with values greater than the threshold
        num_genes_above_threshold = np.sum(counts > th, axis=1)

        # Calculate the proportion of genes above the threshold for each sample
        total_genes = counts.shape[1]
        proportion_above_threshold = num_genes_above_threshold / total_genes

        # Select samples (rows) where the proportion of genes above the threshold is >= prop
        selected_samples = proportion_above_threshold >= prop

        # Update the AnnData object to keep only the selected samples (rows)
        adata_filtered = adata[selected_samples, :].copy()  # Keep all genes, filter rows

        # Calculate total counts per gene for the filtered samples
        gene_counts = np.sum(adata_filtered.X, axis=0)

        # Update the .var DataFrame with total counts
        adata_filtered.var["total_counts"] = gene_counts

        # Update the dictionary with the filtered AnnData object
        anndata_dict[cell_type] = adata_filtered


# Filtering genes


def filter_genes_by_celltype(anndata_dict, gene_lists):
    """
    Filters out genes from AnnData objects based on provided lists of genes to exclude.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with cell types as keys and AnnData objects as values.
    gene_lists : dict[str, list[str]]
        Dictionary with cell types as keys and lists of genes to exclude.

    Returns
    -------
    None
        The function modifies the input AnnData objects in place.
    """

    for cell_type, adata in anndata_dict.items():
        if cell_type in gene_lists:
            genes_to_exclude = gene_lists[cell_type]

            # Check if the genes to exclude are present in the AnnData object
            if any(gene not in adata.var.index for gene in genes_to_exclude):
                print(f"Warning: Some genes to exclude are not found in the AnnData object for {cell_type}.")

            # Create a mask to filter out the specified genes
            mask = ~adata.var.index.isin(genes_to_exclude)

            # Update the AnnData object to keep only the non-excluded genes
            adata_filtered = adata[:, mask].copy()

            # Update the dictionary with the filtered AnnData object
            anndata_dict[cell_type] = adata_filtered

        else:
            print(f"No gene list provided for {cell_type}")


# Function to identify not highly variable genes (HVGs) for exclusion


def get_hvgs(anndata_dict):
    """
    Identify genes to exclude for each AnnData object, based on HVG masking.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with view/cell-type keys and AnnData objects as values.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with cell types as keys and lists of highly variable genes to exclude.
    """

    genes_to_exclude = {}

    for cell_type, adata in anndata_dict.items():
        # Compute HVGs
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=None, inplace=True)

        # Get the HVGs
        mask = ~adata.var["highly_variable"]
        exclude = adata.var.index[mask].tolist()

        genes_to_exclude[cell_type] = exclude

    return genes_to_exclude


# Main function to keep HVGs from all views


def filter_hvgs(anndata_dict):
    """
    Filter genes in AnnData objects by retaining only highly variable genes (HVGs).

    For each AnnData object in the dictionary, highly variable genes are identified
    using Scanpy's ``highly_variable_genes`` method. Genes that are not marked as
    highly variable are removed. After filtering, HVG-related annotation columns
    are dropped from ``.var``.

    This function modifies the input dictionary in place.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with view or cell-type names as keys and AnnData objects as values.

    Returns
    -------
    None
        The input AnnData objects are updated in place.
    """

    gene_lists = get_hvgs(anndata_dict)

    filter_genes_by_celltype(anndata_dict, gene_lists)

    for _x, y in anndata_dict.items():
        y.var = y.var.drop(["highly_variable", "means", "dispersions", "dispersions_norm"], axis=1)
