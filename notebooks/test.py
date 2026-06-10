from pathlib import Path

from config import PLOTS_DIR
from core.sr_study_analysis import run_analysis, run_selected_configs, ExperimentSpec
from core.sr_sim import StudyType

from core.dgp import DGP_EXAMPLES, GARCHProcess, ARGARCHProcess


DGP_EXAMPLES["ar_garch"]=lambda: ARGARCHProcess()
#DGP_EXAMPLES["garch"]=lambda: GARCHProcess(omega=0.05, alpha=0.08, beta=0.72, dist="normal")
DGP_EXAMPLES["garch"]=lambda: GARCHProcess(omega=0.05, alpha=0.1, beta=0.85, dist="normal")

# name : (dgp_names, model_short_names)
scenarios = {
    "normal_hac": (
        ["iid_normal"],
        ["iid_normal", "hac"],
    ),
    "tails_hac": (
        ["iid_t6"],
        ["iid_student_t", "hac", "iid_normal"],
    ),
    "skew_hac": (
        ["iid_skewt60_m05", "iid_skewt6_m05"],
        ["iid_nonnormal", "hac", "iid_normal"],
    ),
    "serial_hac": (
        ["ar1_06_normal", "ar1_m06_normal"],
        ["ar1_normal", "hac", "iid_normal"],
    ),
    "hetero_tails_hac": (
        ["garch"],
        ["garch11", "hac", "iid_normal"],
    ),
    "serial_hetero_hac": (
        ["ar_garch"],
        ["ar1_garch11", "hac", "iid_normal"],
    ),
}

parameters = {
    "n_sim": [1_000, 10_000, 30_000],
    "T":     [500, 1_000, 2_000],
}


N_SIM  = 50_000
N_JOBS = -1
param_name = "T"
param_values = parameters[param_name]

experiments = {
    "normal_hac": ExperimentSpec(# T_default: int = 500, for all
        scenario   = scenarios["normal_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        n_jobs = N_JOBS,
    ),
    "tails_hac": ExperimentSpec(
        scenario   = scenarios["tails_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        n_jobs = N_JOBS,
    ),
    "skew_hac": ExperimentSpec(
        scenario   = scenarios["skew_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        n_jobs = N_JOBS,
    ),
    "serial_hac": ExperimentSpec(
        scenario   = scenarios["serial_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        n_jobs = N_JOBS,
    ),
    "hetero_tails_hac": ExperimentSpec(
        scenario   = scenarios["hetero_tails_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        th_moments = True,
        n_jobs = N_JOBS,
    ),
    "serial_hetero_hac": ExperimentSpec(
        scenario   = scenarios["serial_hetero_hac"],
        param_name = param_name,
        param_values = param_values,
        study_type = StudyType.TWO_SIDED_COVERAGE,
        calib_sigma   = 1.0,
        n_default  = N_SIM,
        th_moments = True,
        n_jobs = N_JOBS,
    ),
}

run_selected_configs(
    experiments,
    selected_experiments=[
        #"normal_hac",
        #"tails_hac",
        #"skew_hac",
        #"serial_hac",
        "hetero_tails_hac",
        #"serial_hetero_hac",
    ],
)