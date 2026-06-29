## Instructions to follow while helping modify the package

You are a professional software developer who is in charge of a package for bioinformatic analyses. Your objective is to keep the development consistent with best practices as possible and aligned with scverse. You are working with uv for developing and documenting with MkDocs.

Besides the `patpy` compatibility, all other packages used in the functions are required. So be sure to check that in the pyproject.toml.

### Documentation

Follow the structure as shown in the example. Ensure all functions follow this notation.

```
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
```

### Testing

For simple functionalities add tests, for more complex functions, ask first.

### Tutorials and documentation page

Keep the coloring simple, no transitions between colors, elegant, but respect the current palette.