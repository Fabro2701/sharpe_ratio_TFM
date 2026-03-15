import numpy as np
import pytest
from core.dgp import ARProcess, ARGARCHProcess, IIDProcess, NormalInnov, StudentTInnov
from core.synth import TrajectorySpec, SyntheticGenerator


@pytest.fixture(scope="module")
def synthetic_data():
    specs = [
        TrajectorySpec(
            IIDProcess(StudentTInnov(df=5.5).calibrate_params(mu=1.5, sigma=1.2)),
            "iid_student", n=10, length=10000
        ),
        TrajectorySpec(
            ARProcess(phi=[0.8], drift=2, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4),
            "ar1_normal", n=10, length=10000
        ),
        TrajectorySpec(
            IIDProcess(NormalInnov().calibrate_params(mu=0.5, sigma=2.0)),
            "iid_normal", n=10, length=10000
        ),
    ]
    data = SyntheticGenerator(seed=42).generate(specs)
    return data, specs


@pytest.mark.parametrize("spec_idx,dgp_name", [
    (0, "iid_student"),
    (1, "ar1_normal"),
    (2, "iid_normal"),
])
def test_per_trajectory_mean(synthetic_data, spec_idx, dgp_name):
    data, specs = synthetic_data
    dgp = specs[spec_idx].dgp
    theo_mean = dgp.get_theo_moments()["mean"]

    traj_means = (
        data.loc[dgp_name]
        .groupby("traj_id")["value"]
        .mean()
    )

    for traj_id, sample_mean in traj_means.items():
        assert abs(sample_mean - theo_mean) < 0.05, (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_mean={sample_mean:.4f}, theo_mean={theo_mean:.4f}, "
            f"mismatch={abs(sample_mean - theo_mean):.4f}"
        )


@pytest.mark.parametrize("spec_idx,dgp_name", [
    (0, "iid_student"),
    (1, "ar1_normal"),
    (2, "iid_normal"),
])
def test_per_trajectory_std(synthetic_data, spec_idx, dgp_name):
    data, specs = synthetic_data
    dgp = specs[spec_idx].dgp
    theo_sigma = dgp.get_theo_moments()["sigma"]

    traj_stds = (
        data.loc[dgp_name]
        .groupby("traj_id")["value"]
        .std()
    )

    for traj_id, sample_std in traj_stds.items():
        assert abs(sample_std - theo_sigma) < 0.05, (
            f"[{dgp_name}] traj_id={traj_id}: "
            f"sample_std={sample_std:.4f}, theo_sigma={theo_sigma:.4f}, "
            f"mismatch={abs(sample_std - theo_sigma):.4f}"
        )