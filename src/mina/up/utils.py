# Dependencies


# This function saves the raw counts in a layer called 'raw_counts' for each AnnData object in a dictionary
def save_raw_counts(anndata_dict, layer_name="raw_counts"):
    """
    Save the current count data (adata.X) into a specified layer for each AnnData in the dict.

    Parameters
    ----------
    - anndata_dict (dict): Dictionary with view/cell-type keys and AnnData objects as values.
    - layer_name (str): The name of the layer to store the raw counts. Default is 'raw_counts'.

    Returns
    -------
    - None: The function modifies the input dictionary in place.
    """
    for _view_name, adata in anndata_dict.items():
        # Save the current count matrix (adata.X) to a new layer
        adata.layers[layer_name] = adata.X.copy()

    # Optionally: Print confirmation that the raw counts have been saved
    print(f"Raw counts saved in the '{layer_name}' layer for each AnnData object.")


def append_view_to_var(anndata_dict, join=":"):
    """
    Prefix feature names in each AnnData with its dict key and `join` separator.

    This modifies the AnnData objects in-place. For example, if the key is "CM"
    and a gene is "gene1", the new var name becomes "CM:gene1" when join=":".

    Parameters
    ----------
    anndata_dict : dict[str, AnnData]
        Dictionary with view/cell-type keys and AnnData objects as values.
    join : str, default ":"
        String used between key and original feature name.

    Returns
    -------
    None
        The function updates `.var_names` of each AnnData in the input dict.
    """
    for key, adata in anndata_dict.items():
        # Ensure string feature names and prefix with key
        new_var = [f"{key}{join}{str(v)}" for v in adata.var_names]
        adata.var_names = new_var
