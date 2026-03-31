import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mina.up import filt


def _make_adata(x, obs_names, var_names, obs_extra=None):
    obs = pd.DataFrame(index=pd.Index(obs_names, dtype=str))
    if obs_extra:
        for key, values in obs_extra.items():
            obs[key] = values
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    return ad.AnnData(X=x, obs=obs, var=var)


def test_filter_anndata_by_ncells_scalar_threshold_dense():
    adata = _make_adata(
        np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        ["d1", "d2", "d3"],
        ["g1", "g2"],
        obs_extra={"psbulk_cells": [5, 20, 20]},
    )
    views = {"A": adata}

    filt.filter_anndata_by_ncells(views, min_cells=20)

    assert list(views["A"].obs_names) == ["d2", "d3"]
    assert views["A"].n_vars == 2
    np.testing.assert_array_equal(views["A"].var["total_counts"].to_numpy(), np.array([8.0, 10.0]))


def test_filter_anndata_by_ncells_dict_threshold_sparse():
    adata = _make_adata(
        sparse.csr_matrix(np.array([[1, 0], [2, 3], [4, 5]], dtype=float)),
        ["d1", "d2", "d3"],
        ["g1", "g2"],
        obs_extra={"psbulk_cells": [10, 11, 12]},
    )
    views = {"A": adata}

    filt.filter_anndata_by_ncells(views, min_cells={"A": 11})

    assert list(views["A"].obs_names) == ["d2", "d3"]
    np.testing.assert_array_equal(views["A"].var["total_counts"].to_numpy(), np.array([6.0, 8.0]))


def test_filter_anndata_by_ncells_missing_threshold_key_raises():
    views = {
        "A": _make_adata(
            np.array([[1.0]]),
            ["d1"],
            ["g1"],
            obs_extra={"psbulk_cells": [1]},
        )
    }

    with pytest.raises(KeyError, match="Key 'A' not found"):
        filt.filter_anndata_by_ncells(views, min_cells={"B": 1})


def test_filter_anndata_by_ncells_missing_psbulk_cells_prints(capsys):
    original = _make_adata(np.array([[1.0, 2.0]]), ["d1"], ["g1", "g2"])
    views = {"A": original}

    filt.filter_anndata_by_ncells(views, min_cells=1)

    captured = capsys.readouterr()
    assert "psbulk_cells" in captured.out
    assert views["A"] is original


def test_filter_views_by_samples_removes_small_views():
    views = {
        "small": _make_adata(np.array([[1.0, 2.0]]), ["d1"], ["g1", "g2"]),
        "ok": _make_adata(np.array([[1.0], [2.0]]), ["d1", "d2"], ["g1"]),
    }

    filt.filter_views_by_samples(views, min_rows=2)

    assert "small" not in views
    assert "ok" in views


def test_filter_genes_byexpr_filters_columns_and_updates_counts(monkeypatch):
    adata = _make_adata(
        np.array([[5, 0, 1], [5, 1, 0], [5, 0, 0]], dtype=float),
        ["d1", "d2", "d3"],
        ["g1", "g2", "g3"],
    )
    views = {"A": adata}

    def fake_cpm_cutoff(lib_size, min_count):
        return np.array([1.0, 1.0, 1.0])

    def fake_cpm(counts, lib_size):
        return np.array(
            [
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

    monkeypatch.setattr(filt.dc.pp.anndata, "_cpm_cutoff", fake_cpm_cutoff)
    monkeypatch.setattr(filt.dc.pp.anndata, "_cpm", fake_cpm)

    filt.filter_genes_byexpr(views, min_count=1, min_prop=2 / 3)

    assert list(views["A"].var_names) == ["g1"]
    np.testing.assert_array_equal(views["A"].var["total_counts"].to_numpy(), np.array([15.0]))
    assert views["A"].n_obs == 3


def test_filter_genes_byexpr_clips_negative_min_count(monkeypatch):
    adata = _make_adata(
        np.array([[0, 0], [1, 0]], dtype=float),
        ["d1", "d2"],
        ["g1", "g2"],
    )
    views = {"A": adata}

    def fake_cpm_cutoff(lib_size, min_count):
        assert min_count == 0
        return np.array([0.0, 0.0])

    def fake_cpm(counts, lib_size):
        return np.array([[1.0, 1.0], [1.0, 1.0]])

    monkeypatch.setattr(filt.dc.pp.anndata, "_cpm_cutoff", fake_cpm_cutoff)
    monkeypatch.setattr(filt.dc.pp.anndata, "_cpm", fake_cpm)

    filt.filter_genes_byexpr(views, min_count=-5, min_prop=0.5)

    assert list(views["A"].var_names) == ["g1", "g2"]


def test_filter_views_by_genes_removes_insufficient_views():
    views = {
        "small": _make_adata(np.array([[1.0]]), ["d1"], ["g1"]),
        "ok": _make_adata(np.array([[1.0, 2.0]]), ["d1"], ["g1", "g2"]),
    }

    filt.filter_views_by_genes(views, min_genes_per_view=2)

    assert "small" not in views
    assert "ok" in views


def test_filter_samples_by_coverage_scalar():
    adata = _make_adata(
        np.array([[1, 0, 1], [0, 0, 1], [5, 5, 5]], dtype=float),
        ["d1", "d2", "d3"],
        ["g1", "g2", "g3"],
    )
    views = {"A": adata}

    filt.filter_samples_by_coverage(views, threshold=0, min_prop=2 / 3)

    assert list(views["A"].obs_names) == ["d1", "d3"]
    np.testing.assert_array_equal(views["A"].var["total_counts"].to_numpy(), np.array([6.0, 5.0, 6.0]))


def test_filter_samples_by_coverage_dict_params():
    views = {
        "A": _make_adata(
            np.array([[1, 0], [1, 1]], dtype=float),
            ["d1", "d2"],
            ["g1", "g2"],
        ),
        "B": _make_adata(
            np.array([[0, 1], [1, 1]], dtype=float),
            ["e1", "e2"],
            ["g1", "g2"],
        ),
    }

    filt.filter_samples_by_coverage(views, threshold={"A": 0, "B": 0}, min_prop={"A": 1.0, "B": 0.5})

    assert list(views["A"].obs_names) == ["d2"]
    assert list(views["B"].obs_names) == ["e1", "e2"]


def test_filter_samples_by_coverage_missing_dict_key_raises():
    views = {"A": _make_adata(np.array([[1.0]]), ["d1"], ["g1"])}

    with pytest.raises(KeyError, match="threshold"):
        filt.filter_samples_by_coverage(views, threshold={"B": 0}, min_prop=0.5)

    with pytest.raises(KeyError, match="proportion"):
        filt.filter_samples_by_coverage(views, threshold=0, min_prop={"B": 0.5})


def test_filter_genes_by_celltype_excludes_genes_and_warns(capsys):
    views = {
        "A": _make_adata(np.array([[1, 2, 3]], dtype=float), ["d1"], ["g1", "g2", "g3"]),
        "B": _make_adata(np.array([[1, 2]], dtype=float), ["e1"], ["g1", "g2"]),
    }

    filt.filter_genes_by_celltype(views, gene_lists={"A": ["g2", "g_missing"]})

    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "No gene list provided for B" in captured.out
    assert list(views["A"].var_names) == ["g1", "g3"]
    assert list(views["B"].var_names) == ["g1", "g2"]


def test_filter_smpls_by_nview_filters_donors_across_views(monkeypatch):
    monkeypatch.setattr(filt, "pd", pd)
    views = {
        "A": _make_adata(np.array([[1.0], [2.0], [3.0]]), ["d1", "d2", "d3"], ["g1"]),
        "B": _make_adata(np.array([[4.0], [5.0]]), ["d1", "d2"], ["g1"]),
        "C": _make_adata(np.array([[6.0], [7.0]]), ["d1", "d4"], ["g1"]),
    }

    filt.filter_smpls_by_nview(views, min_views=2)

    assert list(views["A"].obs_names) == ["d1", "d2"]
    assert list(views["B"].obs_names) == ["d1", "d2"]
    assert list(views["C"].obs_names) == ["d1"]


def test_get_hvgs_without_groupby(monkeypatch):
    adata = _make_adata(np.array([[1, 2, 3], [3, 2, 1]], dtype=float), ["d1", "d2"], ["g1", "g2", "g3"])
    views = {"A": adata}

    def fake_hvg(a, **kwargs):
        a.var["highly_variable"] = [True, False, True]
        a.var["means"] = [0.0, 0.0, 0.0]
        a.var["dispersions"] = [0.0, 0.0, 0.0]
        a.var["dispersions_norm"] = [0.0, 0.0, 0.0]

    monkeypatch.setattr(filt.sc.pp, "highly_variable_genes", fake_hvg)

    excluded = filt.get_hvgs(views, groupby=None)

    assert excluded == {"A": ["g2"]}


def test_get_hvgs_with_groupby_and_ngroups_cut(monkeypatch):
    adata = _make_adata(
        np.array([[1, 2, 3], [3, 2, 1]], dtype=float),
        ["d1", "d2"],
        ["g1", "g2", "g3"],
        obs_extra={"batch": ["b1", "b2"]},
    )
    views = {"A": adata}

    def fake_hvg(a, **kwargs):
        a.var["highly_variable_nbatches"] = [2, 1, 3]
        a.var["dispersions_norm"] = [0.5, 0.9, 0.1]
        a.var["highly_variable"] = [False, False, False]

    monkeypatch.setattr(filt.sc.pp, "highly_variable_genes", fake_hvg)

    excluded = filt.get_hvgs(views, groupby="batch", ngroups_cut=2)

    assert excluded == {"A": ["g2"]}


def test_get_hvgs_with_groupby_raises_when_no_multibatch_hvgs(monkeypatch):
    adata = _make_adata(
        np.array([[1, 2], [3, 4]], dtype=float),
        ["d1", "d2"],
        ["g1", "g2"],
        obs_extra={"batch": ["b1", "b2"]},
    )
    views = {"A": adata}

    def fake_hvg(a, **kwargs):
        a.var["highly_variable_nbatches"] = [1, 1]
        a.var["dispersions_norm"] = [0.1, 0.2]
        a.var["highly_variable"] = [False, False]

    monkeypatch.setattr(filt.sc.pp, "highly_variable_genes", fake_hvg)

    with pytest.raises(ValueError, match="No highly variable genes found"):
        filt.get_hvgs(views, groupby="batch", ngroups_cut=2)


def test_filter_hvgs_filters_and_drops_hvg_columns(monkeypatch):
    views = {
        "A": _make_adata(np.array([[1, 2, 3]], dtype=float), ["d1"], ["g1", "g2", "g3"]),
    }
    views["A"].var["highly_variable"] = [True, True, True]
    views["A"].var["means"] = [0.0, 0.0, 0.0]
    views["A"].var["dispersions"] = [0.0, 0.0, 0.0]
    views["A"].var["dispersions_norm"] = [0.0, 0.0, 0.0]

    def fake_get_hvgs(anndata_dict, groupby=None, ngroups_cut=None):
        return {"A": ["g2"]}

    monkeypatch.setattr(filt, "get_hvgs", fake_get_hvgs)

    filt.filter_hvgs(views)

    assert list(views["A"].var_names) == ["g1", "g3"]
    for col in ["highly_variable", "means", "dispersions", "dispersions_norm"]:
        assert col not in views["A"].var.columns

