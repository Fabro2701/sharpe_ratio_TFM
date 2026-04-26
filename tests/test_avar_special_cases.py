import pytest

# Import the REGISTRY directly from your module
from core.models import REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# Test Data Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Range of realistic and edge-case Sharpe ratios to test
SR_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]

# Define the mathematical reductions.
# Format: (General Model, Special Model, General Kwargs, Special Kwargs)
# The test will assert that: 
# General_Model(sr, **General_Kwargs) == Special_Model(sr, **Special_Kwargs)
MODEL_RELATIONSHIPS = [
    # --- IID Reductions ---
    ("iid_student_t", "iid_normal", {"exc_kurt": 0.0}, {}),
    ("iid_nonnormal", "iid_student_t", {"skew": 0.0, "exc_kurt": 1.5}, {"exc_kurt": 1.5}),
    ("iid_nonnormal", "iid_normal", {"skew": 0.0, "exc_kurt": 0.0}, {}),
    
    # --- AR(1) Reductions ---
    ("ar1_normal", "iid_normal", {"rho": 0.0}, {}),
    ("ar1_nonnormal", "iid_nonnormal", {"rho": 0.0, "skew": 0.5, "exc_kurt": 1.2}, {"skew": 0.5, "exc_kurt": 1.2}),
    ("ar1_nonnormal", "ar1_normal", {"rho": 0.3, "skew": 0.0, "exc_kurt": 0.0}, {"rho": 0.3}),
    
    # --- GARCH(1,1) Reductions ---
    ("garch11", "iid_nonnormal", {"alpha": 0.0, "beta": 0.0, "skew": 0.4, "exc_kurt": 2.0}, {"skew": 0.4, "exc_kurt": 2.0}),
    
    # --- AR(1)-GARCH(1,1) Normal Reductions ---
    ("ar1_garch11normal", "ar1_normal", {"rho": 0.25, "alpha": 0.0, "beta": 0.0}, {"rho": 0.25}),
    ("ar1_garch11normal", "garch11", {"rho": 0.0, "alpha": 0.1, "beta": 0.8}, {"alpha": 0.1, "beta": 0.8, "skew": 0.0, "exc_kurt": 0.0}),
    
    # --- AR(1)-GARCH(1,1) Symmetric Reductions ---
    ("ar1_garch11symm", "ar1_nonnormal", {"rho": 0.3, "alpha": 0.0, "beta": 0.0, "exc_kurt": 1.5}, {"rho": 0.3, "skew": 0.0, "exc_kurt": 1.5}),
    ("ar1_garch11symm", "garch11", {"rho": 0.0, "alpha": 0.1, "beta": 0.8, "exc_kurt": 1.5}, {"alpha": 0.1, "beta": 0.8, "skew": 0.0, "exc_kurt": 1.5}),
]

# ─────────────────────────────────────────────────────────────────────────────
# The Parameterized Test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sr", SR_VALUES)
@pytest.mark.parametrize(
    "general_name, special_name, gen_kwargs, spec_kwargs", 
    MODEL_RELATIONSHIPS,
    ids=[f"{g}->{s}" for g, s, _, _ in MODEL_RELATIONSHIPS] # Creates clean test names in pytest output
)
def test_model_reductions(sr, general_name, special_name, gen_kwargs, spec_kwargs):
    """
    Tests that complex models correctly reduce to simpler models 
    when parameter constraints are applied.
    """
    # Fetch models directly from the imported REGISTRY
    general_model = REGISTRY[general_name]
    special_model = REGISTRY[special_name]

    # Calculate asymptotic variance for both scenarios
    val_general = general_model.avar(sr, **gen_kwargs)
    val_special = special_model.avar(sr, **spec_kwargs)

    # Assert mathematical equivalence
    assert val_general == pytest.approx(val_special), (
        f"Reduction failed at SR={sr}.\n"
        f"General ({general_name} with {gen_kwargs}) -> {val_general}\n"
        f"Special ({special_name} with {spec_kwargs}) -> {val_special}"
    )