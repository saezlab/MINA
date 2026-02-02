# Dependencies:
from __future__ import annotations

import warnings
from collections.abc import Iterable

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData
from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multitest import multipletests

from mina.down.utils import split_by_view


# Funcomics to multiviews


def run_ulm_per_view(
    view_dict: dict[str, pd.DataFrame], net: pd.DataFrame, **kwargs
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run decoupler.mt.ulm on each view's data matrix.

    Parameters
    ----------
    view_dict : dict of {view_name: expression_df}
        Dictionary of expression matrices (e.g. archetypes × genes).
    net : pd.DataFrame
        decoupler-compatible prior knowledge network.
    **kwargs :
        Additional keyword arguments passed to dc.mt.ulm().

    Returns
    -------
    dict of {view_name: {'pw_acts': DataFrame, 'pw_padj': DataFrame}}
    """
    results = {}

    for view, data in view_dict.items():
        print(f"Running ULM for view: {view}")
        pw_acts, pw_padj = dc.mt.ulm(data=data, net=net, **kwargs)

        results[view] = {"pw_acts": pw_acts, "pw_padj": pw_padj}

    return results


# Associations


# get_associations.py
#
# Associate AnnData .X features with .obs covariates using parametric tests
# and optional mixed models (LMM), with BH FDR correction.
def get_associations(adata, test_variable, test_type=None, random_effect=None):
    """
    Associate features (.X columns) of an AnnData object with an .obs covariate.

    Using:
      - For continuous covariates with no random_effect: Pearson correlation.
      - For categorical covariates with no random_effect: one-way ANOVA (F-test).
      - If random_effect is given: likelihood-ratio test on linear mixed models.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    test_variable : str
        Column name in adata.obs to test against.
    test_type : {'continuous', 'categorical'}, optional
        Type of test_variable. If None, inferred from dtype (numeric -> continuous).
    random_effect : str, optional
        Column name in adata.obs to include as a random effect (groups) in LMM.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns ['feature', 'p_value', 'adj_p_value'], where 'feature'
        is the variable name (adata.var_names), and p-values are raw and BH-adjusted.
    """
    # Extract observation DataFrame
    obs = adata.obs.copy()

    # Infer test type if not provided
    if test_type is None:
        if pd.api.types.is_numeric_dtype(obs[test_variable]):
            test_type = "continuous"
        else:
            test_type = "categorical"

    features = list(adata.var_names)
    results = []

    # Loop over features
    for feat in features:
        # Extract expression
        x = adata[:, feat].X
        if hasattr(x, "toarray"):
            vals = x.toarray().flatten()
        else:
            vals = np.asarray(x).flatten()

        # Build per-feature DataFrame
        df = pd.DataFrame({"value": vals, test_variable: obs[test_variable].values})
        if random_effect is not None:
            df[random_effect] = obs[random_effect].values

        df = df.dropna()

        # Skip if insufficient data
        if df.shape[0] < 3:
            results.append((feat, np.nan, np.nan))
            continue

        try:
            if random_effect is None:
                if test_type == "continuous":
                    # Pearson correlation
                    statval, pval = pearsonr(df["value"], df[test_variable].astype(float))
                else:
                    # One-way ANOVA
                    df[test_variable] = df[test_variable].astype("category")
                    groups = [grp.values for _, grp in df.groupby(test_variable)["value"]]
                    if len(groups) < 2:
                        pval = np.nan
                        statval = np.nan
                    else:
                        statval, pval = f_oneway(*groups)
                results.append((feat, statval, pval))
            else:
                # Mixed models with LRT
                if test_type == "continuous":
                    formula_full = f"value ~ {test_variable}"
                else:
                    formula_full = f"value ~ C({test_variable})"

                # Full model
                md_full = smf.mixedlm(formula_full, df, groups=df[random_effect])
                mdf_full = md_full.fit(reml=False)

                params = mdf_full.tvalues  # Pandas Series: estimates for each fixed effect
                pvals = mdf_full.pvalues  # Pandas Series: p-values for each fixed effect
                statval = params.iloc[1]
                pval = pvals.iloc[1]
                results.append((feat, statval, pval))
        except (IndexError, ValueError, TypeError) as e:
            warnings.warn(f"Error processing feature {feat}: {e}", stacklevel=2)
            results.append((feat, np.nan, np.nan))

    # Compile results and adjust p-values
    results_df = pd.DataFrame(results, columns=["feature", "statistic", "p_value"])
    mask = results_df["p_value"].notnull()
    adj = np.full(results_df.shape[0], np.nan)
    if mask.any():
        _, p_adj, _, _ = multipletests(results_df.loc[mask, "p_value"], method="fdr_bh")
        adj[mask] = p_adj
    results_df["adj_p_value"] = adj

    return results_df


# calc_total_variance.py
#
# From the output of get_associations, calculate the total variance
# explained by each feature across all views. Separated by group


def calc_total_variance(adata, associations_df, pval_thrs=0.05):
    """
    Sum the r2 stored in var per group for factors that pass a significance threshold.

    Parameters
    ----------
    adata : AnnData
        Model anndata containing r2 in var.
    associations_df : DataFrame
        Output from get_associations.
    pval_thrs : float, optional
        Significance threshold for p-values to consider a feature significant.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns ['feature', 'p_value', 'adj_p_value'], where 'feature'
        is the variable name (adata.var_names), and p-values are raw and BH-adjusted.
    """
    expl_var_dict = split_by_view(adata.var.copy())
    p = associations_df.set_index("feature")["adj_p_value"]
    sig_factors = p[p < pval_thrs].index
    total_var_dict = {}
    # For every dataframe inside expl_var_dict, sum the values in the columns

    for data_g, values_df in expl_var_dict.items():
        col_sums = values_df.loc[values_df.index.intersection(sig_factors)].sum(axis=0)
        total_var_dict[data_g] = col_sums

    return total_var_dict

# Multiple tests

def get_pval_matrix(adata, covars):
    """
    Compute adjusted p-value associations for multiple covariates in a model AnnData.

    For each covariate, this function calls `down.get_associations` to test its
    association with model factors and collects the adjusted p-values into a
    DataFrame (p_df). Each column corresponds to a covariate and each row to a
    factor (adata.var index).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing model results, with factor information in `.var`
        and covariates in `.obs`.
    covars : list of str
        List of covariate names (columns in `adata.obs`) to test for association.

    Returns
    -------
    p_df : pd.DataFrame
        DataFrame of adjusted p-values with shape (n_factors, n_covariates),
        where rows correspond to factors (`adata.var.index`) and columns
        correspond to tested covariates.
    """
    # Validate covariates
    existing_covars = [c for c in covars if c in adata.obs.columns]
    missing_covars = [c for c in covars if c not in adata.obs.columns]

    if missing_covars:
        warnings.warn(f"Skipping missing covariates not found in adata.obs: {missing_covars}", UserWarning)

    if not existing_covars:
        raise ValueError("None of the provided covariates are present in adata.obs.")

    # Collect adjusted p-values per covariate
    p_df = pd.DataFrame()
    for covar in existing_covars:
        assocs = get_associations(
            adata=adata,
            test_variable=covar,
            test_type=None,
            random_effect=None
        )
        p_df[covar] = assocs["adj_p_value"]

    # Assign factor names as index
    p_df.index = adata.var.index

    return p_df


# Multicellular information networks


def get_loading_gset(col, source_base: str, percentile: float = 0.85) -> pd.DataFrame:
    """
    col: Series or 1-col DataFrame of loadings (index = targets/features)
    source_base: e.g. "Cardiomyocytes"
    percentile: quantile in [0,1] computed on ``values`` within each sign
    """
    s = col.squeeze().dropna()  # ensure Series

    # positives
    pos = s[s > 0]
    if len(pos):
        thr_pos = pos.abs().quantile(percentile)
        pos_keep = pos[pos.abs() >= thr_pos].sort_values(key=lambda x: x.abs(), ascending=False)
        df_pos = pd.DataFrame({"source": f"{source_base}_pos", "target": pos_keep.index, "weight": pos_keep.values})
    else:
        df_pos = pd.DataFrame(columns=["source", "target", "weight"])

    # negatives
    neg = s[s < 0]
    if len(neg):
        thr_neg = neg.abs().quantile(percentile)
        neg_keep = neg[neg.abs() >= thr_neg].sort_values(key=lambda x: x.abs(), ascending=False)
        df_neg = pd.DataFrame({"source": f"{source_base}_neg", "target": neg_keep.index, "weight": neg_keep.values})
    else:
        df_neg = pd.DataFrame(columns=["source", "target", "weight"])

    return pd.concat([df_pos, df_neg], ignore_index=True)


def build_info_networks(
    multicell_scores: pd.DataFrame,
    random_effect: pd.Series | pd.Index | pd.Categorical | np.ndarray | None = None,
    standardize: bool = False,
    drop_na: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit pairwise linear models among columns of an enriched-score matrix to infer directed edges.

    Parameters
    ----------
    multicell_scores : pd.DataFrame
        Enriched scores (samples × features) used as both predictors and targets.
    random_effect : vector-like, optional
        Random intercept grouping vector (e.g., study per sample). Length must match rows.
        If provided, fits LMM; otherwise fits OLS.
    standardize : bool, default False
        If True, z-score each column before fitting.
    drop_na : bool, default True
        If True, drop rows with any NA among target/predictors and (if present) random effect.
    verbose : bool, default True
        If True, warn on skipped fits due to insufficient data.

    Returns
    -------
    pd.DataFrame
        Columns: [target, predictor, coef, R2, cor_estimate, n_samples, model_type]
    """
    results = []

    Xmat = multicell_scores.copy()
    if standardize:
        Xmat = (Xmat - Xmat.mean()) / Xmat.std(ddof=0)
        Xmat = Xmat.replace([np.inf, -np.inf], np.nan)

    group = None
    use_mixed = False
    if random_effect is not None:
        group = pd.Series(random_effect, index=Xmat.index, name="group")
        use_mixed = True

    cols = list(Xmat.columns)

    # Fits the model for each target

    for target in cols:
        preds = [c for c in cols if c != target]
        # Create modeling frame
        data = pd.concat([Xmat[[target] + preds], group] if use_mixed else [Xmat[[target] + preds]], axis=1)

        if drop_na:
            data = data.dropna(axis=0, how="any")

        if data.shape[0] < 3 or len(preds) < 1:
            if verbose:
                warnings.warn(f"{target}: insufficient data after NA handling; skipping.", stacklevel=2)
            continue

        y = data[target].astype(float)
        X = sm.add_constant(data[preds], has_constant="add")

        # Fits the model

        if use_mixed:
            # Random intercept only
            res = sm.MixedLM(endog=y, exog=X, groups=data["group"]).fit(method="lbfgs", reml=True, disp=False)
            model_type = "LMM"
            # Calculating marginal R2 for mixed model
            var_resid = res.scale
            var_random_effect = float(res.cov_re.iloc[0])
            var_fixed_effect = res.predict(X).var()
            total_var = var_fixed_effect + var_random_effect + var_resid
            marginal_r2 = var_fixed_effect / total_var
            R2 = marginal_r2
            # Extract fixed-effect coefficients (exclude intercept)
            coef = res.params.reindex(X.columns).drop("const", errors="ignore")
        else:
            ols = sm.OLS(y, X).fit()
            model_type = "OLS"
            R2 = float(ols.rsquared)
            coef = ols.params.reindex(X.columns).drop("const", errors="ignore")

        n = int(len(y))

        for predictor, estimate in coef.items():
            results.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "coef": float(estimate),
                    "R2": float(R2),
                    "cor_estimate": float(estimate) * float(R2) if np.isfinite(R2) else np.nan,
                    "n_samples": n,
                    "model_type": model_type,
                }
            )

    tidy = pd.DataFrame(results)
    # Optional ordering
    if not tidy.empty:
        tidy = tidy.sort_values(["target", "predictor"]).reset_index(drop=True)
    return tidy


def get_multicell_net(
    test_model: ad.AnnData,
    sel_factor: str,
    random_effect: pd.Series | pd.Index | pd.Categorical | np.ndarray | None = None,
    standardize: bool = False,
    drop_na: bool = True,
    verbose: bool = True,
    percentile: float = 0.85
) -> dict[str, pd.DataFrame]:
    """
    PLACEHOLDER

    Parameters
    ----------
    sel_factor
        Factor name to select from the model (e.g., "Factor1").
    random_effect
        Optional random intercept grouping vector (e.g., 'study' per sample).
        Length must match the number of samples. If omitted, OLS is used.
    standardize
        If True, z-score each column within condition before fitting.
    drop_na
        If True, drop rows with any NA among target/predictors and (if present) random effect.
        If False, models encountering NA will be skipped.
    verbose
        If True, print/skips with warnings on failures.
    percentile
        Percentile threshold in [0,1] to select top genes from loadings per view.

    Returns
    -------
    tidy : dict
        A dictionary where keys are direction and values are DataFrames containing the networks
    """
    # First, extract top genes for each factor
    # Get the gene loadings from the test model
    W = test_model.varm["gene_loadings"]
    model_cols = list(test_model.uns["gene_loadings_columns"])
    factor_names = test_model.var_names.astype(str).tolist()
    loadings_dict = pd.DataFrame(W, columns=model_cols, index=factor_names)
    loadings_dict = split_by_view(loadings_dict)

    gset_dict = {}
    # Now, compute the top genes for each factor
    for vname, _gl in loadings_dict.items():
        # Select the top genes for the specified factor
        net = get_loading_gset(col=loadings_dict[vname].T[[sel_factor]], source_base=vname, percentile=percentile)
        # Extract the pseudobulk information
        data = pd.DataFrame(
            test_model.obsm[vname], columns=test_model.uns[f"{vname}_columns"], index=test_model.obs_names
        )
        data.fillna(0, inplace=True)
        # Enrich the loadings in the data
        pw_acts, pw_padj = dc.mt.ulm(data=data, net=net)
        gset_dict[vname] = pw_acts

    # Concatenate the enrichment results into a single DataFrame
    mcell_scores = pd.concat(gset_dict, axis=1, join="outer")
    mcell_scores.columns = mcell_scores.columns.get_level_values(1)
    # Separate by direction
    pos_cols = mcell_scores.columns[mcell_scores.columns.str.endswith("_pos")]
    neg_cols = mcell_scores.columns[mcell_scores.columns.str.endswith("_neg")]
    # Getting the enriched info
    mcell_scores_pos = mcell_scores.loc[:, pos_cols]
    mcell_scores_neg = mcell_scores.loc[:, neg_cols]

    # Infer multicellular information networks

    neg_net = build_info_networks(
        mcell_scores_neg,
        random_effect=random_effect,  # No random effects in this example
        standardize=standardize,
        drop_na=drop_na,
        verbose=verbose,
    )

    pos_net = build_info_networks(
        mcell_scores_pos,
        random_effect=random_effect,  # No random effects in this example
        standardize=standardize,
        drop_na=drop_na,
        verbose=verbose,
    )

    # Remove the _pos or _neg suffix from the predictor names and target names
    neg_net["target"] = neg_net["target"].str.replace("_neg", "", regex=False)
    neg_net["predictor"] = neg_net["predictor"].str.replace("_neg", "", regex=False)

    pos_net["target"] = pos_net["target"].str.replace("_pos", "", regex=False)
    pos_net["predictor"] = pos_net["predictor"].str.replace("_pos", "", regex=False)

    # Create a dictionary with positive and negative networks
    net_dict = {"positive": pos_net, "negative": neg_net}

    return net_dict


# Projections


def multiview_to_wide(
    views: dict[str, AnnData],  # anndata, typically called anndata_dict in other functions
    sample_key: str | None = None,
    *,
    prefix_features: bool = True,
    return_dataframe: bool = True,
) -> tuple[pd.DataFrame | np.ndarray, pd.Index, list[str]]:
    """
    Build a dense wide matrix (samples × features) from a dict of per-view AnnData.

    Uses the UNION of samples in first-seen order; rows missing in a view are zero-filled.
    """
    if not views:
        raise ValueError("`views` is empty.")

    # 1) Union of sample IDs (first-seen order)
    arrays = []
    for _, av in views.items():
        ids = av.obs_names.astype(str).values if sample_key is None else av.obs[sample_key].astype(str).values
        arrays.append(ids)
    all_ids = np.concatenate(arrays)
    unique_ids = pd.unique(all_ids)

    sample_index = pd.Index(unique_ids, name="sample")
    n_union = len(sample_index)

    blocks: list[np.ndarray] = []
    colnames: list[str] = []

    # 2) For each view, zero-pad to the union of samples and collect feature blocks
    for vname, av in views.items():
        # ids in this view
        if sample_key is None:
            ids_v = pd.Index(av.obs_names.astype(str), name="sample")
        else:
            if sample_key not in av.obs:
                raise KeyError(f"sample_key '{sample_key}' not found in obs for view '{vname}'.")
            ids_v = pd.Index(av.obs[sample_key].astype(str).values, name="sample")

        # guard: no duplicate IDs inside a view
        if ids_v.duplicated().any():
            dups = ids_v[ids_v.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate sample IDs in view '{vname}' under key '{sample_key}': {dups[:5]}...")

        # map union ids -> row position in this view (or None)
        pos_map = pd.Series(np.arange(av.n_obs), index=ids_v).to_dict()
        src_pos = [pos_map.get(sid, None) for sid in sample_index]

        # dense feature matrix
        Xv = av.X
        if sp.issparse(Xv):
            Xv = Xv.toarray()
        else:
            Xv = np.asarray(Xv)

        # feature names (optionally prefix with view name)
        raw_feats = av.var_names.astype(str).tolist()
        feats = [f"{vname}:{f}" if (prefix_features and not f.startswith(f"{vname}:")) else f for f in raw_feats]

        # zero-padded block
        block = np.zeros((n_union, Xv.shape[1]), dtype=Xv.dtype)
        present = [(i_t, i_s) for i_t, i_s in enumerate(src_pos) if i_s is not None]
        if present:
            tgt_idx, src_idx = zip(*present, strict=False)
            block[np.fromiter(tgt_idx, int), :] = Xv[np.fromiter(src_idx, int), :]

        blocks.append(block)
        colnames.extend(feats)

    # 3) Horizontal stack → one wide matrix
    wide = np.hstack(blocks) if len(blocks) > 1 else blocks[0]

    if return_dataframe:
        wide_df = pd.DataFrame(wide, index=sample_index, columns=colnames)
        return wide_df
    else:
        return wide, sample_index, colnames


def project_wide_to_factors(
    wide: pd.DataFrame | np.ndarray,
    W: np.ndarray,
    model_cols: Iterable[str],
    factor_names: Iterable[str] | None = None,
    rcond: float | None = None,
    center: bool = False,
    sample_annotations: pd.DataFrame | None = None,
) -> ad.AnnData:
    """
    Project a wide samples×features matrix into factor space using the model's loadings.

    Parameters
    ----------
    wide : pd.DataFrame
        (n_samples × n_features_in_wide). Columns are feature names (strings) that
        overlap with `model_cols`.
    W : np.ndarray
        Loadings matrix with shape (n_factors × n_features_total) from the model.
    model_cols : Iterable[str]
        Feature names (len = n_features_total) giving the EXACT column order of W.
    factor_names : Optional[Iterable[str]]
        Names for output factor columns. If None, uses range(n_factors) for arrays or DF columns.
    rcond : Optional[float]
        rcond passed to np.linalg.pinv (smaller → keeps more singular values). If None, numpy default.
    center : bool
        If True, center each column (subtract mean) after aligning to the model's features.
    sample_annotations : Optional[pd.DataFrame]
        Optional sample-level annotations to join into the returned AnnData.obs.

    Returns
    -------
    AnnData
        An AnnData with X = projected factor scores (n_samples × n_factors) and
        optional annotations joined into .obs.
    """
    # -- 0) Normalize inputs
    if isinstance(wide, pd.DataFrame):
        X = wide.values
        feat_all = wide.columns.astype(str).to_numpy()
        idx = wide.index
    else:
        raise ValueError("Provide wide dataframe of projected data")

    model_cols = list(model_cols)
    n_factors, n_feats_model = W.shape
    if n_feats_model != len(model_cols):
        raise ValueError("W width != len(model_cols).")

    # -- 1) Intersect features with the model
    model_pos = {f: i for i, f in enumerate(model_cols)}
    present_mask = np.array([f in model_pos for f in feat_all], dtype=bool)

    if present_mask.sum() == 0:
        raise ValueError("No overlapping features between input and model features.")

    # Subset input to shared features
    X_shared = X[:, present_mask]
    shared_feats = feat_all[present_mask]

    # -- 2) Reorder shared features to match the model's column order
    order_in_model = np.fromiter((model_pos[f] for f in shared_feats), dtype=int, count=len(shared_feats))
    sort_idx = np.argsort(order_in_model)
    X_shared_sorted = X_shared[:, sort_idx]
    shared_feats_sorted = shared_feats[sort_idx]

    # -- 3) Optional centering
    if center:
        col_means = X_shared_sorted.mean(axis=0)
        X_shared_sorted = X_shared_sorted - col_means

    # -- 4) Compute or use pseudoinverse
    W_pinv = np.linalg.pinv(W, rcond=rcond)

    # Select only the rows (features) we kept, in model order
    sel = np.fromiter((model_pos[f] for f in shared_feats_sorted), dtype=int, count=len(shared_feats_sorted))
    W_pinv_shared = W_pinv[sel, :]  # (n_shared_features × n_factors)

    # -- 5) Project
    S = X_shared_sorted @ W_pinv_shared  # (n_samples × n_factors)

    # -- 6) Return
    cols = list(factor_names) if factor_names is not None else [f"Factor{i + 1}" for i in range(n_factors)]
    index = idx if idx is not None else pd.RangeIndex(X.shape[0])
    proj = pd.DataFrame(S, index=index, columns=cols)
    proj_ad = sc.AnnData(proj)
    if sample_annotations is not None:
        ann = sample_annotations.copy()
        proj_ad.obs = proj_ad.obs.join(ann, how="left", on="sample")

    return proj_ad
