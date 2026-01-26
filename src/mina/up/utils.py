# Dependencies


# This function saves the raw counts in a layer called 'raw_counts' for each AnnData object in a dictionary
def save_raw_counts(anndata_dict, layer_name="raw_counts"):
    """
    Save the current count data (adata.X) into a specified layer for each AnnData in the dict.

    Parameters
    ----------
    - anndata_dict : anndata_dict 
        Dictionary with view/cell-type keys and AnnData objects as values.
    - layer_name : str 
        The name of the layer to store the raw counts. Default is 'raw_counts'.

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



def merge_adata_views(
    studies: list[dict[str, AnnData]],
    view_mode: str = "union",
    min_view_studies: int = 2,
    var_mode: str = "outer",
    min_var_studies: int = 2
) -> dict[str, AnnData]:
    """
    Merge multiple AnnData dictionaries (studies) into a single dictionary.

    Parameters
    ----------
    studies : list[dict[str, AnnData]]
        List of dictionaries, each with view/cell-type keys and AnnData objects as values.
    view_mode : str, ["union", "intersection", "min_n"], default "union"
        Determines how the view's merging is handled.
        "union": all views present in the studies
        "intersection": only views present in all studies
        "min_n": views present in at least min_view_studies
    min_view_studies : int,  default 2
        threshold for "min_n" in view_mode
    var_mode : str, ["inner", "outer", "min_n"], default "outer"
        Determines how the variables's merging is handled.
        "inner": only views present in all studies
        "outer": all variables present in the studies
        "min_n": variables present in at least min_var_studies

    Assumptions
    -----------
    - .obs columns are already harmonized across studies
    - .obs is merged by strict intersection
    - .obs_names are unique across studies
    - .var_names are harmonized across studies
    - views are harmonized accross studies

    Returns
    -------
    merged : dict[str, AnnData]
        Dictionary of merged AnnData objects, one per retained view.

        Keys
        ----
        Each key corresponds to a view (modality/cell type) retained 
        according to `view_mode` across the input studies.

        Values
        ------
        Each value is an AnnData object resulting from concatenating
        the corresponding AnnData objects from all studies that contain 
        that view. Guarantees:

        - `.obs` columns: only columns present in all contributing studies
          are retained (strict intersection).
        - `.obs_names` (row identifiers): all original observation names 
          are preserved; duplicates across studies are not allowed.
        - `.var` columns (features):
            * "inner" → only variables present in all contributing studies
            * "outer" → all variables present in at least one contributing study
            * "min_n" → variables present in at least `min_var_studies` studies
        - `.uns` and other metadata are merged conservatively with unique keys.
        - The resulting AnnData objects are copies; modifying them will
          not affect the original input studies.       
    """

    if view_mode == "min_n" and min_view_studies < 2:
        raise ValueError("min_view_studies must be >= 2")

    if var_mode == "min_n" and min_var_studies < 2:
        raise ValueError("min_var_studies must be >= 2")

    n_studies = len(studies)

    view_counts = {}
    for study in studies:
        for view in study.keys():
            view_counts[view] = view_counts.get(view, 0) + 1

    if view_mode == "intersection":
        keep_views = {v for v, c in view_counts.items() if c == n_studies}
    elif view_mode == "union":
        keep_views = set(view_counts.keys())
    elif view_mode == "min_n":
        keep_views = {v for v, c in view_counts.items() if c >= min_view_studies}
    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")

    view_to_adatas = {}
    for view in keep_views:
        view_to_adatas[view] = []

    for study in studies:
        for view in keep_views:
            if view in study:
                view_to_adatas[view].append(study[view])

    merged = {}

    for view, adatas in view_to_adatas.items():
        if len(adatas) == 1:
            merged_adata = adatas[0].copy()
        else:
            # Concatenate first using outer join on variables
            merged_adata = ad.concat(
                adatas,
                axis=0,
                join="outer",
                merge="unique"
            )

            if var_mode in {"inner", "min_n"}:
                var_counts = {}
                for a in adatas:
                    for v in a.var_names:
                        var_counts[v] = var_counts.get(v, 0) + 1

                if var_mode == "inner":
                    keep_vars = {
                        v for v, c in var_counts.items() if c == len(adatas)
                    }
                else:  # min_n
                    keep_vars = {
                        v for v, c in var_counts.items() if c >= min_var_studies
                    }

                merged_adata = merged_adata[:, sorted(keep_vars)]

        obs_cols = None
        for a in adatas:
            if obs_cols is None:
                obs_cols = set(a.obs.columns)
            else:
                obs_cols &= set(a.obs.columns)

        merged_adata.obs = merged_adata.obs.loc[:, sorted(obs_cols)]

        merged[view] = merged_adata

    return merged
