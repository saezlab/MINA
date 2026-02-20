import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mina.up import pp, utils


def _make_adata(x, obs_names, var_names, obs_extra=None):
    obs = pd.DataFrame(index=pd.Index(obs_names, dtype=str))
    if obs_extra:
        for k, v in obs_extra.items():
            obs[k] = v
    var = pd.DataFrame(index=pd.Index(var_names))
    return ad.AnnData(X=np.array(x, dtype=float), obs=obs, var=var)


def test_extract_metadata_from_obs_keeps_only_stable_columns():
    obs = pd.DataFrame(
        {
            "donor_id": ["patient_2", "patient_10", "patient_2", "patient_10"],
            "age": [30, 40, 30, 40],
            "sex": [np.nan, "F", np.nan, "F"],
            "unstable": [1, 1, 2, 1],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id", sort=False)

    assert set(metadata.columns) == {"donor_id", "age", "sex"}
    assert "unstable" not in metadata.columns
    assert metadata.index.name is None
    assert list(metadata["donor_id"]) == ["patient_10", "patient_2"]


def test_extract_metadata_from_obs_sort_natural_order():
    obs = pd.DataFrame(
        {
            "donor_id": ["patient_10", "patient_2", "patient_10", "patient_2"],
            "site": ["A", "B", "A", "B"],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id", sort=True)

    assert list(metadata["donor_id"]) == ["patient_2", "patient_10"]


def test_extract_metadata_from_obs_warns_if_no_stable_columns(capsys):
    obs = pd.DataFrame(
        {
            "donor_id": ["d1", "d1", "d2", "d2"],
            "value": [1, 2, 3, 4],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id")

    captured = capsys.readouterr()
    assert "No stable columns found" in captured.out
    assert list(metadata.columns) == ["donor_id"]


def test_split_anndata_by_celltype_splits_and_returns_copies():
    pdata = _make_adata(
        x=[[1, 2], [3, 4], [5, 6]],
        obs_names=["c1", "c2", "c3"],
        var_names=["g1", "g2"],
        obs_extra={"cell_type": ["T", "B", "T"]},
    )

    split = pp.split_anndata_by_celltype(pdata, grouping="cell_type")

    assert set(split.keys()) == {"T", "B"}
    assert list(split["T"].obs_names) == ["c1", "c3"]
    assert list(split["B"].obs_names) == ["c2"]
    split["T"].X[0, 0] = 999
    assert pdata.X[0, 0] == 1


def test_split_anndata_by_celltype_missing_grouping_raises():
    pdata = _make_adata(x=[[1]], obs_names=["c1"], var_names=["g1"])

    with pytest.raises(ValueError, match="not found"):
        pp.split_anndata_by_celltype(pdata, grouping="cell_type")


def test_norm_log_calls_scanpy_pipeline_center_true(monkeypatch, capsys):
    views = {
        "A": _make_adata([[1, 2]], ["c1"], ["g1", "g2"]),
        "B": _make_adata([[3, 4]], ["c2"], ["g1", "g2"]),
    }
    calls = []

    def fake_normalize_total(adata, target_sum, exclude_highly_expressed):
        calls.append(("normalize_total", target_sum, exclude_highly_expressed, adata.n_obs))

    def fake_log1p(adata):
        calls.append(("log1p", adata.n_obs))

    def fake_scale(adata, max_value):
        calls.append(("scale", max_value, adata.n_obs))

    monkeypatch.setattr(pp.sc.pp, "normalize_total", fake_normalize_total)
    monkeypatch.setattr(pp.sc.pp, "log1p", fake_log1p)
    monkeypatch.setattr(pp.sc.pp, "scale", fake_scale)

    pp.norm_log(views, target_sum=1234, exclude_highly_expressed=True, max_value=5.0, center=True)

    assert [c[0] for c in calls].count("normalize_total") == 2
    assert [c[0] for c in calls].count("log1p") == 2
    assert [c[0] for c in calls].count("scale") == 2
    assert ("normalize_total", 1234, True, 1) in calls
    assert ("scale", 5.0, 1) in calls
    assert "scaling complete" in capsys.readouterr().out


def test_norm_log_skips_scale_when_center_false(monkeypatch, capsys):
    views = {"A": _make_adata([[1, 2]], ["c1"], ["g1", "g2"])}
    calls = []

    monkeypatch.setattr(pp.sc.pp, "normalize_total", lambda *args, **kwargs: calls.append("normalize_total"))
    monkeypatch.setattr(pp.sc.pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(pp.sc.pp, "scale", lambda *args, **kwargs: calls.append("scale"))

    pp.norm_log(views, center=False)

    assert calls == ["normalize_total", "log1p"]
    assert "Normalization and log-transformation complete" in capsys.readouterr().out


def test_save_raw_counts_creates_independent_layer(capsys):
    adata = _make_adata([[1, 2], [3, 4]], ["c1", "c2"], ["g1", "g2"])
    views = {"A": adata}

    utils.save_raw_counts(views)

    np.testing.assert_array_equal(views["A"].layers["raw_counts"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    views["A"].X[0, 0] = 99
    assert views["A"].layers["raw_counts"][0, 0] == 1.0
    assert "Raw counts saved" in capsys.readouterr().out


def test_save_raw_counts_custom_layer_name():
    adata = _make_adata([[1]], ["c1"], ["g1"])
    views = {"A": adata}

    utils.save_raw_counts(views, layer_name="counts_backup")

    assert "counts_backup" in views["A"].layers


def test_append_view_to_var_prefixes_names_and_supports_custom_separator():
    adata = _make_adata([[1, 2]], ["c1"], [1, 2])
    views = {"Tcell": adata}

    utils.append_view_to_var(views, join="__")

    assert list(views["Tcell"].var_names) == ["Tcell__1", "Tcell__2"]


def _studies_for_merge():
    study1 = {
        "A": _make_adata(
            [[1, 2], [3, 4]],
            ["s1_a1", "s1_a2"],
            ["g1", "g2"],
            obs_extra={"donor": ["d1", "d2"], "age": [50, 60]},
        ),
        "B": _make_adata(
            [[9, 8]],
            ["s1_b1"],
            ["g2", "g3"],
            obs_extra={"donor": ["d1"], "age": [50]},
        ),
    }
    study2 = {
        "A": _make_adata(
            [[5, 6], [7, 8]],
            ["s2_a1", "s2_a2"],
            ["g2", "g3"],
            obs_extra={"donor": ["d3", "d4"], "age": [40, 41], "extra": ["x", "y"]},
        ),
        "C": _make_adata(
            [[10]],
            ["s2_c1"],
            ["g5"],
            obs_extra={"donor": ["d3"], "age": [40]},
        ),
    }
    return study1, study2


def test_merge_adata_views_validates_arguments():
    s1, s2 = _studies_for_merge()

    with pytest.raises(ValueError, match="same length"):
        utils.merge_adata_views([s1, s2], ["one"])

    with pytest.raises(ValueError, match="Unknown view_mode"):
        utils.merge_adata_views([s1, s2], ["one", "two"], view_mode="bad_mode")

    with pytest.raises(ValueError, match="min_view_studies must be >= 2"):
        utils.merge_adata_views([s1, s2], ["one", "two"], view_mode="min_n", min_view_studies=1)

    with pytest.raises(ValueError, match="min_var_studies must be >= 2"):
        utils.merge_adata_views([s1, s2], ["one", "two"], var_mode="min_n", min_var_studies=1)


def test_merge_adata_views_union_outer_and_obs_column_intersection():
    s1, s2 = _studies_for_merge()

    merged = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="union", var_mode="outer")

    assert set(merged.keys()) == {"A", "B", "C"}
    assert list(merged["A"].var_names) == ["g1", "g2", "g3"]
    assert list(merged["A"].obs.columns) == ["age", "donor", "study"]
    assert set(merged["A"].obs["study"]) == {"study1", "study2"}


def test_merge_adata_views_intersection_and_min_n_view_modes():
    s1, s2 = _studies_for_merge()

    merged_intersection = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="intersection")
    merged_min_n = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="min_n", min_view_studies=2)

    assert set(merged_intersection.keys()) == {"A"}
    assert set(merged_min_n.keys()) == {"A"}


def test_merge_adata_views_inner_and_min_n_var_modes():
    s1, s2 = _studies_for_merge()

    merged_inner = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="intersection", var_mode="inner")
    merged_min_n = utils.merge_adata_views(
        [s1, s2], ["study1", "study2"], view_mode="intersection", var_mode="min_n", min_var_studies=2
    )

    assert list(merged_inner["A"].var_names) == ["g2"]
    assert list(merged_min_n["A"].var_names) == ["g2"]


def test_merge_adata_views_returns_copies():
    s1, s2 = _studies_for_merge()

    merged = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="union")

    merged["B"].X[0, 0] = 999
    assert s1["B"].X[0, 0] == 9

