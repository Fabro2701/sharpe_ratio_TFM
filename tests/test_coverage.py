import pytest

from core.dgp import IIDProcess, NormalInnov, StudentTInnov, ARProcess
from core.models import (
    IIDNormalModel,
    IIDStudentTModel,
    IIDNonNormalModel,
    AR1NormalModel,
    AR1NonNormalModel,
    REGISTRY,
)
from core.coverage import run_coverage_study, DGPSpec

# ─────────────────────────────────────────────────────────────────────────────
# Study parameters
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05
TARGET_COVERAGE = 1.0 - ALPHA
THRESHOLD = 0.01
T = 500
N_SIM = 10000
TARGET_SR = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# DGP registry
# ─────────────────────────────────────────────────────────────────────────────

DGP_REGISTRY: dict[str, DGPSpec] = {
    "iid_normal":  DGPSpec(IIDProcess(NormalInnov()),                       "iid_normal"),
    "iid_student": DGPSpec(IIDProcess(StudentTInnov(df=6)),                 "iid_student"),
    "ar1_normal":  DGPSpec(ARProcess(phi=0.6, innov=NormalInnov()),         "ar1_normal"),
    "ar1_student": DGPSpec(ARProcess(phi=0.6, innov=StudentTInnov(df=6)),   "ar1_student"),
}


def make_pair(dgp_name: str, model_name: str) -> tuple[str, DGPSpec, object]:
    return (
        f"{dgp_name}__{model_name}",
        DGP_REGISTRY[dgp_name],
        REGISTRY[model_name],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pair definitions  (dgp_name, model_name)
# ─────────────────────────────────────────────────────────────────────────────

GOOD_PAIRS = [make_pair(*p) for p in [
    ("iid_normal",  "iid_normal"),
    ("iid_normal",  "iid_student_t"),
    ("iid_student", "iid_student_t"),
    ("iid_normal",  "iid_nonnormal"),
    ("iid_student", "iid_nonnormal"),
    ("iid_normal",  "ar1_normal"),
    #("iid_student", "ar1_normal"),
    ("ar1_normal",  "ar1_normal"),
    ("iid_normal",  "ar1_nonnormal"),
    ("iid_student", "ar1_nonnormal"),
    ("ar1_normal",  "ar1_nonnormal"),
    ("ar1_student", "ar1_nonnormal"),
]]

BAD_PAIRS = [make_pair(*p) for p in [
    ("ar1_normal",  "iid_normal"),    # AR structure ignored
    ("ar1_student", "iid_student_t"),   # AR structure ignored
    ("iid_student", "iid_normal"),    # heavy tails ignored
]]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_single(dgp_spec: DGPSpec, model, seed: int = 42) -> float:
    res = run_coverage_study(
        dgp_specs=[dgp_spec],
        avar_models=[model],
        target_sr=TARGET_SR,
        T=T,
        n_sim=N_SIM,
        alpha=ALPHA,
        th_moments=False,
        seed=seed,
        verbose=False,
        n_jobs=8
    )
    return res.coverage[0]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair_id,dgp_spec,model", GOOD_PAIRS, ids=[p[0] for p in GOOD_PAIRS])
def test_coverage_within_threshold(pair_id, dgp_spec, model):
    empirical_coverage = _run_single(dgp_spec, model)

    assert abs(empirical_coverage - TARGET_COVERAGE) <= THRESHOLD, (
        f"[{pair_id}] coverage out of bounds:\n"
        f"  empirical={empirical_coverage:.4f}, "
        f"  target={TARGET_COVERAGE:.4f}, "
        f"  diff={abs(empirical_coverage - TARGET_COVERAGE):.4f}, "
        f"  threshold={THRESHOLD:.4f}"
    )


@pytest.mark.parametrize("pair_id,dgp_spec,model", BAD_PAIRS, ids=[p[0] for p in BAD_PAIRS])
def test_coverage_bad_model_fails_threshold(pair_id, dgp_spec, model):
    empirical_coverage = _run_single(dgp_spec, model)

    assert abs(empirical_coverage - TARGET_COVERAGE) > THRESHOLD, (
        f"[{pair_id}] mismatched model unexpectedly achieved good coverage:\n"
        f"  empirical={empirical_coverage:.4f}, "
        f"  target={TARGET_COVERAGE:.4f}, "
        f"  diff={abs(empirical_coverage - TARGET_COVERAGE):.4f}, "
        f"  threshold={THRESHOLD:.4f}"
    )