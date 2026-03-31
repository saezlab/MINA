# Dependencies
from anndata import AnnData
import anndata as ad
import decoupler as dc
import pandas as pd

# This function saves the raw counts in a layer called 'raw_counts' for each AnnData object in a dictionary
def save_raw_counts(anndata_dict, layer_name="raw_counts"):
    """
    Store raw count data in a layer for each AnnData object.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary of AnnData objects.
    layer_name : str
        Name of the layer used to store raw counts.

    Returns
    -------
    None
        The input dictionary is modified in place.
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
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping views to AnnData objects.
    join : str
        Separator used between view name and feature name.
        Default is ":"

    Returns
    -------
    None
        Updates ``.var_names`` in place.
    """
    for key, adata in anndata_dict.items():
        # Ensure string feature names and prefix with key
        new_var = [f"{key}{join}{str(v)}" for v in adata.var_names]
        adata.var_names = new_var



def merge_adata_views(
    studies: list[dict[str, AnnData]],
    study_names: list[str],
    view_mode: str = "union",
    min_view_studies: int = 2,
    var_mode: str = "outer",
    min_var_studies: int = 2
) -> dict[str, AnnData]:
    """
    Merge multiple study-level AnnData dictionaries into unified views.

    Parameters
    ----------
    studies : list[dict[str, anndata.AnnData]]
        List of study dictionaries, each mapping view names to AnnData objects.
    study_names : list[str]
        Unique identifiers for each study. Must align with ``studies``.
    view_mode : ``{'union', 'intersection', 'min_n'}``
        Strategy for selecting views across studies.
    min_view_studies : int
        Minimum number of studies required when ``view_mode='min_n'``.
    var_mode : ``{'inner', 'outer', 'min_n'}``
        Strategy for merging variables (features).
    min_var_studies : int
        Minimum number of studies required when ``var_mode='min_n'``.

    Assumptions
    -----------
    note::
        Observation columns are harmonized across studies.
        Observation names are unique across studies.
        Feature names are harmonized across studies.
        View names are consistent across studies.
        ``study_names`` uniquely identify studies.


    Returns
    -------
    merged : dict[str, anndata.AnnData]
        Dictionary of merged AnnData objects, one per retained view.

        **Keys**
            Each key corresponds to a view (modality/cell type) retained
            according to ``view_mode`` across the input studies.

        **Values**
            Each value is an AnnData object resulting from concatenating
            the corresponding AnnData objects from all studies that contain 
            that view. Guarantees:

            - `.obs` columns: only columns present in all contributing studies
              are retained (strict intersection).
            - `.obs_names` (row identifiers): all original observation names 
              are preserved; duplicates across studies are not allowed.
            - `.obs["study"]`: column indicating the study of origin for each
              observation, using the names provided in ``study_names``.
            - `.var` columns (features):
                * ``"inner"`` → only variables present in all contributing studies
                * ``"outer"`` → all variables present in at least one contributing study
                * ``"min_n"`` → variables present in at least ``min_var_studies`` studies
            - `.uns` and other metadata are merged conservatively with unique keys.
            - The resulting AnnData objects are copies; modifying them will 
              not affect the original input studies.
    """

    if len(studies) != len(study_names):
        raise ValueError("study_names must have the same length as studies")

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


    view_to_adatas = {view: [] for view in keep_views}

    for study_name, study in zip(study_names, studies):
        for view in keep_views:
            if view in study:
                a = study[view].copy()
                a.obs["study"] = study_name
                view_to_adatas[view].append(a)

    merged = {}

    for view, adatas in view_to_adatas.items():

        if len(adatas) == 1:
            merged_adata = adatas[0].copy()
        else:
            merged_adata = ad.concat(
                adatas,
                axis=0,
                join="outer",
                merge="unique"
            )

        if len(adatas) > 1:

            var_counts = {}
            for a in adatas:
                for v in a.var_names:
                    var_counts[v] = var_counts.get(v, 0) + 1

            first_var_order = list(adatas[0].var_names)

            if var_mode == "inner":
                keep_vars_set = {v for v, c in var_counts.items() if c == len(adatas)}

            elif var_mode == "min_n":
                keep_vars_set = {v for v, c in var_counts.items() if c >= min_var_studies}

            elif var_mode == "outer":
                keep_vars_set = set(var_counts.keys())

            else:
                raise ValueError(f"Unknown var_mode: {var_mode}")

            ordered_vars = [v for v in first_var_order if v in keep_vars_set]
            for a in adatas[1:]:
                for v in a.var_names:
                    if v in keep_vars_set and v not in ordered_vars:
                        ordered_vars.append(v)

            merged_adata = merged_adata[:, ordered_vars]

            var_frames = []
            for a in adatas:
                df = a.var.copy()
                df = df.loc[df.index.intersection(ordered_vars)]
                var_frames.append(df)

            combined_var = (
                pd.concat(var_frames)
                .loc[~pd.concat(var_frames).index.duplicated(keep="first")]
            )

            # Reindex to ensure full ordered_vars coverage
            combined_var = combined_var.reindex(ordered_vars)

            merged_adata.var = combined_var


        obs_cols = None
        for a in adatas:
            if obs_cols is None:
                obs_cols = set(a.obs.columns)
            else:
                obs_cols &= set(a.obs.columns)

        ordered_obs_cols = sorted(c for c in obs_cols if c != "study")

        if "study" in obs_cols:
            ordered_obs_cols.append("study")

        merged_adata.obs = merged_adata.obs.loc[:, ordered_obs_cols]

        merged[view] = merged_adata

    return merged


def convert_views_to_functions(anndata_dict, net, tmin = 5):
    """
    Applies decoupler's run ulm function to each AnnData object in the input dictionary using the provided network.

    Rewrites the input dictionary in place, replacing each AnnData object with the result of the decoupler analysis.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary with AnnData objects as values.

    net : pandas.DataFrame
        Long-format (tidy) DataFrame representing a network, where each row defines
        an interaction between a source and a target.
        
        Required columns:
        - ``source``: identifier of the source node
        - ``target``: identifier of the target node
        
        Optional columns:
        - ``weight``: numeric value representing interaction strength
        
    tmin : int, default=5
        Minimum number of targets required per source. Sources with fewer than
        ``tmin`` associated targets are filtered out.

    Returns
    -------
    None
        The function modifies the input dictionary in place.
    """

    for cell_type, adata in anndata_dict.items():
        dc.mt.ulm(data=adata, net=net, tmin=tmin)
        score = dc.pp.get_obsm(adata=adata, key="score_ulm")

        # Update the dictionary with the filtered AnnData object
        anndata_dict[cell_type] = score.copy()

def make_membership_matrix(adata, pathways_df, gene_col="genesymbol", pathway_col="pathway"):
    """
    Build a boolean gene × pathway membership matrix aligned with adata.var.index.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with genes in .var.index
    pathways_df : pd.DataFrame
        Long-format DataFrame with at least two columns: gene_col and pathway_col
    gene_col : str
        Column in pathways_df containing gene names
    pathway_col : str
        Column in pathways_df containing pathway names

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame (rows=adata.var.index, columns=unique pathways)
    """
    # Get unique pathways
    pathways = pathways_df[pathway_col].unique()
    genes = adata.var.index

    # Initialize matrix
    membership = pd.DataFrame(False, index=genes, columns=pathways)

    # Fill True where membership exists
    for pw in pathways:
        genes_in_pw = pathways_df.loc[pathways_df[pathway_col] == pw, gene_col]
        membership.loc[membership.index.isin(genes_in_pw), pw] = True

    # Store in .varm
    adata.varm["pathway_membership"] = membership
    adata.uns["pathway_names"] = list(membership.columns)  # optional: keep names

    return membership
