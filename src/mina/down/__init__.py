"""Downstream analysis tools for MINA model outputs."""

from .tl import (
    build_info_networks,
    calc_total_variance,
    get_associations,
    get_loading_gset,
    get_multicell_net,
    get_pval_matrix,
    lr_usage,
    multiview_to_wide,
    project_wide_to_factors,
    run_ulm_per_view,
)
from .utils import (
    model_to_anndata,
    restore_anns_factor,
    split_by_view,
)

__all__ = [
    "run_ulm_per_view",
    "get_associations",
    "calc_total_variance",
    "get_pval_matrix",
    "get_loading_gset",
    "build_info_networks",
    "get_multicell_net",
    "multiview_to_wide",
    "project_wide_to_factors",
    "model_to_anndata",
    "split_by_view",
    "lr_usage",
    "restore_anns_factor",
]
