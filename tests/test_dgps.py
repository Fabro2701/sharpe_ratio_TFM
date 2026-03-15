import numpy as np
import pandas as pd
import pytest
from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth — add new specs here only
# ─────────────────────────────────────────────────────────────────────────────

SPEC_BUILDERS = [
    ("iid_student", lambda: TrajectorySpec(
        IIDProcess(StudentTInnov(df=5.5).calibrate_params(mu=1.5, sigma=1.2)),
        "iid_student", n=10, length=10000
    )),
    ("ar1_normal", lambda: TrajectorySpec(
        ARProcess(phi=[0.8], drift=2, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4),
        "ar1_normal", n=10, length=10000
    )),
    ("iid_normal", lambda: TrajectorySpec(
        IIDProcess(NormalInnov().calibrate_params(mu=0.5, sigma=2.0)),
        "iid_normal", n=10, length=10000
    )),
]

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
def test_per_trajectory_mean(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_mean = dgp.get_theo_moments()["mean"]

    traj_means = data.loc[dgp_name].groupby("traj_id")["value"].mean()

    for traj_id, sample_mean in traj_means.items():
        assert abs(sample_mean - theo_mean) < 0.05, (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_mean={sample_mean:.4f}, theo_mean={theo_mean:.4f}, "
            f"mismatch={abs(sample_mean - theo_mean):.4f}"
        )


@pytest.mark.parametrize("dgp_name,builder", DGP_PARAMS, ids=DGP_IDS)
def test_per_trajectory_std(synthetic_data, dgp_name, builder):
    data, _ = synthetic_data
    dgp = builder().dgp
    theo_sigma = dgp.get_theo_moments()["sigma"]

    traj_stds = data.loc[dgp_name].groupby("traj_id")["value"].std()

    for traj_id, sample_std in traj_stds.items():
        assert abs(sample_std - theo_sigma) < 0.05, (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_std={sample_std:.4f}, theo_sigma={theo_sigma:.4f}, "
            f"mismatch={abs(sample_std - theo_sigma):.4f}"
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