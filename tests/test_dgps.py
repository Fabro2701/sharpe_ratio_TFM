
import pytest

import numpy as np
import pandas as pd
from scipy import stats

from core.dgp import DGP_EXAMPLES
from core.synth import TrajectorySpec, SyntheticGenerator


THRESHOLDS = {
    "mean":0.05,
    "std":0.05,
    "kurt":2.0,
    "skew":1.0,
    "rho":0.05,
}

CALIBRATION_MOMENTS = {
    "iid_normal": {"mu": 0.5, "sigma": 2.0},
    "iid_t6": {"mu": 1.5, "sigma": 1.2},
    "ar1_06_normal": {"mu": 1.5, "sigma": 0.4},
    "ar1_m06_normal": {"mu": 1.5, "sigma": 0.4},
    "ar1_06_t6": {"mu": 1.5, "sigma": 0.4},
    "ar1_m06_t6": {"mu": 1.5, "sigma": 0.4},
    "garch_normal": {"mu": 1.5, "sigma": 0.4},
}

SPEC_BUILDERS = []
length = int(1e5)
n_traj = 10
for name, params in CALIBRATION_MOMENTS.items():
    builder = lambda name=name, p=params: TrajectorySpec(
        DGP_EXAMPLES[name]().calibrate_params(**p),
        name, n=n_traj, length=length
    )
    SPEC_BUILDERS.append((name, builder))

DGP_PARAMS = [(dgp_name, builder) for dgp_name, builder in SPEC_BUILDERS]
DGP_IDS    = [dgp_name for dgp_name, _ in SPEC_BUILDERS]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    specs = [builder() for _, builder in SPEC_BUILDERS]
    data  = SyntheticGenerator(seed=42).generate(specs)
    data  = data.sort_index() #for performance
    return data, specs


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_no_nan_values(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data

    traj_ids = data.loc[dgp_name].index.get_level_values("traj_id").unique()
    for traj_id in traj_ids:
        nan_count = data.loc[(dgp_name, traj_id), "value"].isna().sum()
        assert nan_count == 0, (
            f"[{dgp_name}] traj_id={traj_id}: found {nan_count} NaN values"
        )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_theoretical_moments_match_calibration(dgp_name, builder):

    dgp = builder().dgp
    theo_mom = dgp.get_theo_moments()

    calib = CALIBRATION_MOMENTS[dgp_name]

    theo_mean = theo_mom["mean"]
    theo_std  = theo_mom["sigma"]

    target_mean = calib["mu"]
    target_std  = calib["sigma"]

    assert abs(theo_mean - target_mean) < THRESHOLDS["mean"], (
        f"[{dgp_name}] theoretical mean mismatch:\n"
        f"target_mean={target_mean:.4f}, theo_mean={theo_mean:.4f}, "
        f"diff={abs(theo_mean - target_mean):.6f}"
    )

    assert abs(theo_std - target_std) < THRESHOLDS["std"], (
        f"[{dgp_name}] theoretical std mismatch:\n"
        f"target_std={target_std:.4f}, theo_std={theo_std:.4f}, "
        f"diff={abs(theo_std - target_std):.6f}"
    )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_mean_theo(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_mean = dgp.get_theo_moments()["mean"]

    traj_means = data.loc[dgp_name].groupby("traj_id")["value"].mean()

    for traj_id, sample_mean in traj_means.items():
        assert abs(sample_mean - theo_mean) < THRESHOLDS["mean"], (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_mean={sample_mean:.4f}, theo_mean={theo_mean:.4f}, "
            f"mismatch={abs(sample_mean - theo_mean):.4f}"
        )


@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_std_theo(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_sigma = dgp.get_theo_moments()["sigma"]

    traj_stds = data.loc[dgp_name].groupby("traj_id")["value"].std()

    for traj_id, sample_std in traj_stds.items():
        assert abs(sample_std - theo_sigma) < THRESHOLDS["std"], (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_std={sample_std:.4f}, theo_sigma={theo_sigma:.4f}, "
            f"mismatch={abs(sample_std - theo_sigma):.4f}"
        )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_kurt_theo(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_kurt = dgp.get_theo_moments()["exc_kurt"]

    traj_kurt = (
        data.loc[dgp_name]
        .groupby("traj_id")["value"]
        .apply(lambda x: stats.kurtosis(x, fisher=True))
    )

    for traj_id, sample_kurt in traj_kurt.items():
        assert abs(sample_kurt - theo_kurt) < THRESHOLDS["kurt"], (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_exc_kurt={sample_kurt:.4f}, theo_exc_kurt={theo_kurt:.4f}, "
            f"mismatch={abs(sample_kurt - theo_kurt):.4f}"
        )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_skew_theo(synthetic_data, dgp_name, builder):# no skewd gdps yet
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_skew = dgp.get_theo_moments()["skew"]

    traj_skew = (
        data.loc[dgp_name]
        .groupby("traj_id")["value"]
        .apply(lambda x: stats.skew(x))
    )

    for traj_id, sample_skew in traj_skew.items():
        assert abs(sample_skew - theo_skew) < THRESHOLDS["skew"], (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_skew={sample_skew:.4f}, theo_skew={theo_skew:.4f}, "
            f"mismatch={abs(sample_skew - theo_skew):.4f}"
        )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_rho_theo(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_rho = dgp.get_theo_moments()["rho"]

    def fit_rho(x):
        lag = x[:-1]; y = x[1:]
        dm  = lag - lag.mean()
        den = float(np.dot(dm, dm))
        rho = 0.0 if den < 1e-12 else float(
            np.clip(np.dot(dm, y - y.mean()) / den, -0.999, 0.999)
        )
        return rho
    
    traj_rho = (
        data.loc[dgp_name]
        .groupby("traj_id")["value"]
        .apply(lambda x: fit_rho(x))
    )

    for traj_id, sample_rho in traj_rho.items():
        assert abs(sample_rho - theo_rho) < THRESHOLDS["rho"], (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_rho={sample_rho:.4f}, theo_rho={theo_rho:.4f}, "
            f"mismatch={abs(sample_rho - theo_rho):.4f}"
        )

@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_reproducibility_per_traj(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    data_repro = SyntheticGenerator(seed=42).generate([builder()])

    traj_ids = data.loc[dgp_name].index.get_level_values("traj_id").unique()
    for traj_id in traj_ids:
        original   = data.loc[(dgp_name, traj_id), "value"].values
        replicated = data_repro.loc[(dgp_name, traj_id), "value"].values
        assert np.array_equal(original, replicated), (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"first mismatch at index {np.where(original != replicated)[0][0]}"
        )