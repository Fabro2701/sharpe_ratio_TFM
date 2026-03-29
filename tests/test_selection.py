import pytest
import numpy as np
from core.dgp import IIDProcess, NormalInnov, StudentTInnov, ARProcess, GARCHProcess
from core.synth import TrajectorySpec, SyntheticGenerator
from core.model_selection import (
    IIDNormal, IIDStudent, IIDSkewStudent, IIDGeneralizedError,
    AR1Normal, AR1Student, AR1SkewStudent,
    GARCH11Normal, GARCH11Student, GARCH11SkewStudent,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global knobs
# ─────────────────────────────────────────────────────────────────────────────

N_SERIES: int               = 100
SERIES_LENGTH: int          = 2_000
CORRECT_RATE_THRESHOLD: float = 0.80

# ─────────────────────────────────────────────────────────────────────────────
# DGP registry  —  add a new DGP here; reference it by key in DGP_MODEL_PAIRS
# ─────────────────────────────────────────────────────────────────────────────

DGPS: dict[str, callable] = {
    "iid_normal": (
        lambda: IIDProcess(NormalInnov()).calibrate_params(mu=0.5, sigma=2.0)
    ),
    "iid_student": (
        lambda: IIDProcess(StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=1.2)
    ),
    "ar1_06_normal": (
        lambda: ARProcess(phi=0.6, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_m06_normal": (
        lambda: ARProcess(phi=-0.6, innov=NormalInnov()).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_06_t": (
        lambda: ARProcess(phi=0.6, innov=StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "ar1_m06_t": (
        lambda: ARProcess(phi=-0.6, innov=StudentTInnov(df=6)).calibrate_params(mu=1.5, sigma=0.4)
    ),
    "garch_n": (
        lambda: GARCHProcess(mu=0.05, omega=0.05, alpha=0.10, beta=0.85,
                             dist="normal").calibrate_params(mu=1.5, sigma=0.4)
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Model baskets  —  use focused baskets to avoid convergence issues with
# models that are ill-suited for a given DGP
# ─────────────────────────────────────────────────────────────────────────────

BASKETS: dict[str, list] = {
    "iid": [
        IIDNormal(), IIDStudent(), IIDSkewStudent(), IIDGeneralizedError(),
    ],
    "ar1": [
        AR1Normal(), AR1Student(), AR1SkewStudent(),
    ],
    "iid_vs_ar1": [
        IIDNormal(), IIDStudent(),
        AR1Normal(), AR1Student(),
    ],
    "garch": [
        GARCH11Normal(), GARCH11Student(), GARCH11SkewStudent(),
    ],
    "iid_vs_garch": [
        IIDNormal(), IIDStudent(),
        GARCH11Normal(), GARCH11Student(), GARCH11SkewStudent(),
    ],
    "all": [
        IIDNormal(), IIDStudent(), IIDSkewStudent(), IIDGeneralizedError(),
        AR1Normal(), AR1Student(), AR1SkewStudent(),
        GARCH11Normal(), GARCH11Student(), GARCH11SkewStudent(),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Trio list  —  (dgp_id, basket_name, expected_model_short_name)
#
# • dgp_id          : key in DGPS
# • basket_name     : key in BASKETS  (controls which models are fitted)
# • expected_model  : short_name of the model that should win on BIC
#
# The same dgp_id can appear in multiple rows (different baskets / hypotheses).
# The pytest id is built as  <dgp_id>__<basket_name>__<expected_model>  so
# every row is uniquely identifiable even when the dgp_id repeats.
# ─────────────────────────────────────────────────────────────────────────────

DGP_MODEL_PAIRS: list[tuple[str, str, str]] = [
    # dgp_id           basket_name    expected_model
    ("iid_normal",     "iid",         "iid_normal"),
    ("iid_student",    "iid",         "iid_t"),
    ("ar1_06_normal",  "iid_vs_ar1",  "ar1_normal"),
    ("ar1_m06_normal", "iid_vs_ar1",  "ar1_normal"),
    ("ar1_06_t",       "iid_vs_ar1",  "ar1_t"),
    ("ar1_m06_t",      "iid_vs_ar1",  "ar1_t"),
    ("garch_n",        "iid_vs_garch","garch11_normal"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Pytest ids — unique even when the same dgp_id appears in multiple rows
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_IDS = [
    f"{dgp_id}__{basket_name}__{expected}"
    for dgp_id, basket_name, expected in DGP_MODEL_PAIRS
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_bic_model(series: np.ndarray, models: list) -> str:
    """Return the short_name of the model with the lowest BIC on *series*."""
    results = [m.fit(series, rescale=True) for m in models]
    return min(results, key=lambda r: r.bic).model_name


def _generate_series(dgp_id: str, n: int, length: int, seed: int) -> list[np.ndarray]:
    """Draw *n* independent series from the DGP registered under *dgp_id*."""
    dgp = DGPS[dgp_id]()
    spec = TrajectorySpec(dgp, name="_fit_sel_", n=n, length=length)
    data = SyntheticGenerator(seed=seed).generate([spec]).sort_index()

    return [
        data.loc[("_fit_sel_", traj_id), "value"].to_numpy()
        for traj_id in data.loc["_fit_sel_"].index.get_level_values("traj_id").unique()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture — generate all series once per module
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def all_series_cache() -> dict[str, list[np.ndarray]]:
    """
    Keyed by dgp_id; each entry holds N_SERIES arrays of length SERIES_LENGTH.
    A dgp_id that appears in multiple pairs is only generated once.
    """
    unique_dgp_ids = dict.fromkeys(dgp_id for dgp_id, _, _ in DGP_MODEL_PAIRS)
    return {
        dgp_id: _generate_series(dgp_id, n=N_SERIES, length=SERIES_LENGTH, seed=42)
        for dgp_id in unique_dgp_ids
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dgp_id,basket_name,expected_model", DGP_MODEL_PAIRS, ids=_PAIR_IDS)
def test_bic_selects_correct_model(all_series_cache, dgp_id, basket_name, expected_model):
    """
    Fit every model in *basket_name* on N_SERIES draws from *dgp_id* and assert
    that BIC picks *expected_model* in at least CORRECT_RATE_THRESHOLD of cases.
    """
    series_list = all_series_cache[dgp_id]
    models      = BASKETS[basket_name]

    correct = sum(
        _best_bic_model(s, models) == expected_model
        for s in series_list
    )
    correct_rate = correct / len(series_list)

    assert correct_rate >= CORRECT_RATE_THRESHOLD, (
        f"[{dgp_id} | basket={basket_name}] "
        f"BIC selected '{expected_model}' in only {correct}/{len(series_list)} series "
        f"({correct_rate:.1%} < threshold {CORRECT_RATE_THRESHOLD:.1%}).\n"
        f"Series length={SERIES_LENGTH}, N_SERIES={N_SERIES}."
    )