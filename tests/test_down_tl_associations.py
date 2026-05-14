import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr

from mina.down.tl import get_associations


def _make_adata(x, obs, var_names):
    obs = pd.DataFrame(obs, index=[f"s{i}" for i in range(len(next(iter(obs.values()))))])
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    return ad.AnnData(X=np.asarray(x, dtype=float), obs=obs, var=var)


def test_get_associations_continuous_detects_linear_feature():
    dose = np.arange(12, dtype=float)
    alternating_null = np.array([1, -1] * 6, dtype=float)
    adata = _make_adata(
        np.column_stack([dose, alternating_null]),
        {"dose": dose},
        ["dose_aligned", "null"],
    )

    result = get_associations(adata, test_variable="dose", test_type="continuous")

    expected_stat, expected_p = pearsonr(dose, dose)
    null_stat, null_p = pearsonr(alternating_null, dose)
    assert list(result["feature"]) == ["dose_aligned", "null"]
    np.testing.assert_allclose(result["statistic"], [expected_stat, null_stat])
    np.testing.assert_allclose(result["p_value"], [expected_p, null_p])
    assert result.loc[result["feature"] == "dose_aligned", "adj_p_value"].item() < 0.001
    assert result.loc[result["feature"] == "null", "adj_p_value"].item() > 0.5


def test_get_associations_categorical_detects_group_shift():
    group = np.array(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    group_shift = np.array(
        [
            1,
            1.2,
            0.8,
            1.1,
            0.9,
            1.0,
            10,
            10.2,
            9.8,
            10.1,
            9.9,
            10.0,
            20,
            20.2,
            19.8,
            20.1,
            19.9,
            20.0,
        ],
        dtype=float,
    )
    repeated_null = np.array([1, 2, 3, 4, 5, 6] * 3, dtype=float)
    adata = _make_adata(
        np.column_stack([group_shift, repeated_null]),
        {"group": group},
        ["group_shift", "null"],
    )

    result = get_associations(adata, test_variable="group", test_type="categorical")

    expected_stat, expected_p = f_oneway(group_shift[:6], group_shift[6:12], group_shift[12:])
    null_stat, null_p = f_oneway(repeated_null[:6], repeated_null[6:12], repeated_null[12:])
    assert list(result["feature"]) == ["group_shift", "null"]
    np.testing.assert_allclose(result["statistic"], [expected_stat, null_stat])
    np.testing.assert_allclose(result["p_value"], [expected_p, null_p])
    assert result.loc[result["feature"] == "group_shift", "adj_p_value"].item() < 0.001
    assert result.loc[result["feature"] == "null", "adj_p_value"].item() == 1.0


def test_get_associations_mixed_continuous_detects_within_donor_effect():
    rng = np.random.default_rng(42)
    donors = []
    dose = []
    dose_effect = []
    null = []
    for donor in range(12):
        donor_intercept = donor * 5.0
        for value in [0, 1, 2, 3, 4]:
            donors.append(f"d{donor}")
            dose.append(value)
            dose_effect.append(donor_intercept + 3 * value + rng.normal(0, 0.3))
            null.append(donor_intercept + rng.normal(0, 0.3))

    adata = _make_adata(
        np.column_stack([dose_effect, null]),
        {"dose": dose, "donor": donors},
        ["dose_effect", "null"],
    )

    result = get_associations(
        adata,
        test_variable="dose",
        test_type="continuous",
        random_effect="donor",
    )

    assert list(result["feature"]) == ["dose_effect", "null"]
    assert result.loc[result["feature"] == "dose_effect", "statistic"].item() > 100
    assert result.loc[result["feature"] == "dose_effect", "adj_p_value"].item() < 0.001
    assert abs(result.loc[result["feature"] == "null", "statistic"].item()) < 1
    assert result.loc[result["feature"] == "null", "adj_p_value"].item() > 0.5


def test_get_associations_mixed_categorical_detects_within_donor_effect():
    rng = np.random.default_rng(123)
    donors = []
    treatment = []
    treatment_effect = []
    null = []
    for donor in range(12):
        donor_intercept = donor * 5.0
        for value in ["control", "treated", "control", "treated"]:
            donors.append(f"d{donor}")
            treatment.append(value)
            treatment_effect.append(donor_intercept + (4 if value == "treated" else 0) + rng.normal(0, 0.3))
            null.append(donor_intercept + rng.normal(0, 0.3))

    adata = _make_adata(
        np.column_stack([treatment_effect, null]),
        {"treatment": treatment, "donor": donors},
        ["treatment_effect", "null"],
    )

    result = get_associations(
        adata,
        test_variable="treatment",
        test_type="categorical",
        random_effect="donor",
    )

    assert list(result["feature"]) == ["treatment_effect", "null"]
    assert result.loc[result["feature"] == "treatment_effect", "statistic"].item() > 40
    assert result.loc[result["feature"] == "treatment_effect", "adj_p_value"].item() < 0.001
    assert abs(result.loc[result["feature"] == "null", "statistic"].item()) < 1
    assert result.loc[result["feature"] == "null", "adj_p_value"].item() > 0.9
